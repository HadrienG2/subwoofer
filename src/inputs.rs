//! Benchmarking of individual data sources (registers, L1/L2/L3 caches, RAM...)

use common::operations::{self, Benchmark};
use criterion::{measurement::WallTime, BenchmarkGroup};
use rand::prelude::*;

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
        common::inputs::generate_positive(&mut inputs, &mut config.rng, num_subnormals);

        // Name this subnormal configuration
        let input_name = format!(
            "{num_subnormals:0num_digits$}in{INPUT_REGISTERS}",
            // Leading zeros works around poor criterion bench name sorting
            num_digits = INPUT_REGISTERS.ilog10() as usize + 1
        );

        // Run all the benchmarks on this input
        operations::run_benchmark(
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
        common::inputs::generate_positive(input_storage, &mut config.rng, num_subnormals);

        // Name this subnormal configuration
        let input_name = format!(
            // Leading zeros works around poor criterion bench name sorting
            "{:0>5.1}%",
            subnormal_probability * 100.0,
        );

        // Run all the benchmarks on this input
        operations::run_benchmark(
            config.benchmark,
            config.group,
            &mut *input_storage,
            input_name,
            &mut config.rng,
        );
    }
}
