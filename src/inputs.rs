//! Benchmarking of individual data sources (registers, L1/L2/L3 caches, RAM...)

use common::{floats::FloatLike, inputs::FloatSet, operations::Benchmark};
use criterion::{measurement::WallTime, BenchmarkGroup, Throughput};
use rand::prelude::*;
use std::time::Instant;

/// Maximum granularity of subnormal occurence probabilities
///
/// Higher is more precise, but the benchmark execution time in the default
/// configuration is multipled accordingly.
const MAX_SUBNORMAL_CONFIGURATIONS: usize = const {
    if cfg!(feature = "subnormal_freq_resolution_1in128") {
        128 // 0.78125% granularity
    } else if cfg!(feature = "subnormal_freq_resolution_1in64") {
        64 // 1.5625% granularity
    } else if cfg!(feature = "subnormal_freq_resolution_1in32") {
        32 // 3.125% granularity
    } else if cfg!(feature = "subnormal_freq_resolution_1in16") {
        16 // 6.25% granularity
    } else if cfg!(feature = "subnormal_freq_resolution_1in8") {
        8 // 12.5% granularity
    } else {
        4 // 25% granularity
    }
};

/// Configuration for benchmarking a certain data source
pub(crate) struct DataSourceConfiguration<'criterion, 'group, B: Benchmark> {
    /// Random number generator
    pub rng: ThreadRng,

    /// Criterion benchmark group
    pub group: &'group mut BenchmarkGroup<'criterion, WallTime>,

    /// Workload to be exercised on this data
    pub benchmark: B,
}

/// Run a certain benchmark for a certain scalar/SIMD type, using a certain
/// degree of ILP and a certain number of register inputs.
#[cfg(feature = "register_data_sources")]
#[inline(never)] // Faster build + easier profiling
pub(crate) fn benchmark_registers<B: Benchmark, const INPUT_REGISTERS: usize>(
    mut config: DataSourceConfiguration<B>,
) {
    // Iterate over subnormal configurations
    let num_subnormal_configurations = INPUT_REGISTERS.min(MAX_SUBNORMAL_CONFIGURATIONS);
    for subnormal_share in 0..=num_subnormal_configurations {
        // Generate input data
        let num_subnormals = subnormal_share * INPUT_REGISTERS / num_subnormal_configurations;
        let mut inputs = [B::Float::default(); INPUT_REGISTERS];
        generate_positive(&mut inputs, &mut config.rng, num_subnormals);

        // Name this subnormal configuration
        let input_name = format!(
            "{num_subnormals:0num_digits$}in{INPUT_REGISTERS}",
            // Leading zeros works around poor criterion bench name sorting
            num_digits = INPUT_REGISTERS.ilog10() as usize + 1
        );

        // Run all the benchmarks on this input
        run_benchmark(
            config.benchmark,
            config.group,
            inputs,
            input_name,
            &mut config.rng,
        );
    }
}

/// Run a certain benchmark for a certain scalar/SIMD type, using a certain
/// degree of ILP and memory inputs of a certain size.
#[inline(never)] // Faster build + easier profiling
pub(crate) fn benchmark_memory<B: Benchmark>(
    mut config: DataSourceConfiguration<B>,
    input_storage: &mut [B::Float],
) {
    // Iterate over subnormal configurations
    for subnormal_probability in 0..=MAX_SUBNORMAL_CONFIGURATIONS {
        // Generate input data
        let subnormal_probability =
            subnormal_probability as f64 / MAX_SUBNORMAL_CONFIGURATIONS as f64;
        let num_subnormals = (subnormal_probability * input_storage.len() as f64).round() as usize;
        generate_positive(input_storage, &mut config.rng, num_subnormals);

        // Name this subnormal configuration
        let input_name = format!(
            // Leading zeros works around poor criterion bench name sorting
            "{:0>5.1}%",
            subnormal_probability * 100.0,
        );

        // Run all the benchmarks on this input
        run_benchmark(
            config.benchmark,
            config.group,
            &mut *input_storage,
            input_name,
            &mut config.rng,
        );
    }
}

/// Fill a buffer with positive normal and subnormal inputs
///
/// The order of normal vs subnormal inputs is not randomized yet, this will be
/// taken care of by
/// [`make_sequence()`](common::inputs::FloatSet::make_sequence()) later on.
fn generate_positive<T: FloatLike, R: Rng>(set: &mut [T], rng: &mut R, num_subnormals: usize) {
    // Generate subnormal inputs
    assert!(num_subnormals <= set.len());
    let (subnormal_target, normal_target) = set.split_at_mut(num_subnormals);
    let subnormal = T::subnormal_sampler::<R>();
    for target in subnormal_target {
        *target = subnormal(rng);
    }

    // Split normal inputs, if any, in two parts. The smaller half is random...
    if normal_target.is_empty() {
        return;
    }
    let normal = T::normal_sampler::<R>();
    let num_normals = normal_target.len();
    let (random, inverses) = normal_target.split_at_mut(num_normals / 2);
    for elem in random.iter_mut() {
        *elem = normal(rng);
    }

    // ...while the other half is made of the inverses of the random half,
    // padded with an extra 1 when the halves do not have the same length.
    //
    // This ensures that the product of all normal inputs is 1, and thus that
    // multiplicative accumulators get back to their initial value after being
    // multiplied by all normal inputs. Which, in turn, maximizes the chances
    // that multiplicative random walks won't land above T::MAX or below
    // T::MIN_POSITIVE, even when constantly reusing a small set of inputs in a
    // long-running benchmarking loop.
    for (elem, inverse) in random.iter().zip(inverses.iter_mut()) {
        *inverse = T::splat(1.0) / *elem;
    }
    if inverses.len() > random.len() {
        assert_eq!(inverses.len(), random.len() + 1);
        *inverses.last_mut().unwrap() = T::splat(1.0);
    }
}

/// Measure the performance of a [`Benchmark`] on certain inputs
#[inline(never)] // Faster build + easier profiling
pub fn run_benchmark<B: Benchmark>(
    benchmark: B,
    group: &mut BenchmarkGroup<WallTime>,
    mut inputs: impl FloatSet<Element = B::Float>,
    input_name: String,
    mut rng: impl Rng,
) {
    group.throughput(Throughput::Elements(B::num_operations(&inputs) as u64));
    group.bench_function(input_name, move |b| {
        b.iter_custom(|iters| {
            // For each benchmark iteration batch, we ensure that...
            //
            // - The compiler cannot leverage the fact that the initial
            //   accumulators are always the same in order to eliminate
            //   "redundant" computations across iteration batches.
            // - Inputs are randomly reordered from one batch to another, which
            //   will avoid input order-related hardware bias if criterion runs
            //   enough batches (as it does in its default configuration).
            let mut benchmark = benchmark.begin_run(&mut rng);
            let mut inputs = inputs.make_sequence(&mut rng);

            // Timed region, this is the danger zone where inlining and compiler
            // optimizations must be reviewed very carefully.
            let start = Instant::now();
            for _ in 0..iters {
                benchmark.integrate_inputs(&mut inputs);
            }
            let elapsed = start.elapsed();

            // Tell the compiler that the accumulated results are used so it
            // doesn't delete the computation
            for acc in benchmark.accumulators() {
                pessimize::consume::<B::Float>(*acc);
            }
            elapsed
        })
    });
}
