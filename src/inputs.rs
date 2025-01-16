//! Benchmarking of individual data sources (registers, L1/L2/L3 caches, RAM...)

use common::operations::{Benchmark, BenchmarkRun};
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
pub(crate) fn benchmark_registers<B: Benchmark>(
    mut config: DataSourceConfiguration<B>,
    input_registers: usize,
) {
    // Iterate over subnormal configurations
    let num_subnormal_configurations = input_registers.min(MAX_SUBNORMAL_CONFIGURATIONS);
    for subnormal_share in 0..=num_subnormal_configurations {
        // Generate input data
        let num_subnormals = subnormal_share * input_registers / num_subnormal_configurations;
        config.benchmark.setup_inputs(num_subnormals);

        // Name this subnormal configuration
        let input_name = format!(
            "{num_subnormals:0num_digits$}in{input_registers}",
            // Leading zeros works around poor criterion bench name sorting
            num_digits = input_registers.ilog10() as usize + 1
        );

        // Run all the benchmarks on this input
        run_benchmark(
            &mut config.benchmark,
            config.group,
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
    input_len: usize,
) {
    // Iterate over subnormal configurations
    for subnormal_probability in 0..=MAX_SUBNORMAL_CONFIGURATIONS {
        // Generate input data
        let subnormal_probability =
            subnormal_probability as f64 / MAX_SUBNORMAL_CONFIGURATIONS as f64;
        let num_subnormals = (subnormal_probability * input_len as f64).round() as usize;
        config.benchmark.setup_inputs(num_subnormals);

        // Name this subnormal configuration
        let input_name = format!(
            // Leading zeros works around poor criterion bench name sorting
            "{:0>5.1}%",
            subnormal_probability * 100.0,
        );

        // Run all the benchmarks on this input
        run_benchmark(
            &mut config.benchmark,
            config.group,
            input_name,
            &mut config.rng,
        );
    }
}

/// Measure the performance of a [`Benchmark`] on certain inputs
#[inline(never)] // Faster build + easier profiling
pub fn run_benchmark<B: Benchmark>(
    benchmark: &mut B,
    group: &mut BenchmarkGroup<WallTime>,
    input_name: String,
    rng: &mut impl Rng,
) {
    group.throughput(Throughput::Elements(benchmark.num_operations() as u64));
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
            let mut run = benchmark.start_run(rng);

            // Timed region, this is the danger zone where inlining and compiler
            // optimizations must be reviewed very carefully.
            let start = Instant::now();
            for _ in 0..iters {
                run.integrate_inputs();
            }
            let elapsed = start.elapsed();

            // Tell the compiler that the accumulated results are used so it
            // doesn't delete the computation
            for acc in run.accumulators() {
                pessimize::consume(*acc);
            }
            elapsed
        })
    });
}
