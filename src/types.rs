//! Full data type benchmarking

use crate::operations::{self, TypeConfiguration};
use common::floats::FloatLike;
use criterion::Criterion;
use hwlocality::Topology;
use rand::prelude::*;

/// Common setup for fully benchmarking one or more data types
pub struct TypeBenchmark<'criterion> {
    /// Random number generator
    rng: ThreadRng,

    /// Criterion benchmark harness
    criterion: &'criterion mut Criterion,

    /// Name and size in bytes of memory inputs
    memory_input_sizes: Box<[(String, usize)]>,
}
//
impl<'criterion> TypeBenchmark<'criterion> {
    /// Set up a type benchmark harness
    pub fn new(criterion: &'criterion mut Criterion) -> Self {
        // Find out how many bytes of data we can reliably fits in L1, L2, ...
        // and add a dataset size that does not fit in any for completeness
        let cache_stats = Topology::new().unwrap().cpu_cache_stats().unwrap();
        let smallest_data_cache_sizes = cache_stats.smallest_data_cache_sizes();
        let max_size_to_fit = |cache_size: u64| cache_size / 2;
        let min_size_to_overflow = |cache_size: u64| cache_size * 8;
        let memory_input_sizes = smallest_data_cache_sizes
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, cache_size)| (format!("L{}", idx + 1), max_size_to_fit(cache_size)))
            .chain(std::iter::once((
                "RAM".to_string(),
                min_size_to_overflow(*cache_stats.total_data_cache_sizes().last().unwrap()),
            )))
            .take(if cfg!(feature = "more_memory_data_sources") {
                usize::MAX
            } else {
                1
            })
            .map(|(name, size)| (name, usize::try_from(size).unwrap()))
            .collect::<Box<[_]>>();

        // Return common benchmark harness
        Self {
            rng: rand::thread_rng(),
            criterion,
            memory_input_sizes,
        }
    }

    /// Benchmark a certain floating-point type
    ///
    /// In order to work around name sorting limitations in criterion's reports,
    /// SIMD type names like "f32x8" should have their lane count padded with
    /// leading zeros so that they all have the same number of digits.
    #[inline(never)] // Faster build + easier profiling
    pub fn benchmark_type<T: FloatLike>(&mut self, t_name: &'static str) -> &mut Self {
        // Set up memory inputs for this type
        let mut memory_inputs = self
            .memory_input_sizes
            .iter()
            .map(|(name, size)| {
                let num_elems = *size / std::mem::size_of::<T>();
                (
                    name.as_ref(),
                    vec![T::default(); num_elems].into_boxed_slice(),
                )
            })
            .collect::<Box<[_]>>();

        // Benchmark each enabled operation
        operations::benchmark_all(TypeConfiguration {
            rng: self.rng.clone(),
            criterion: self.criterion,
            t_name,
            memory_inputs: &mut memory_inputs,
        });
        self
    }
}
