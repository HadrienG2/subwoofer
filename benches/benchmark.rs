#![feature(portable_simd)]

use common::{
    arch::MIN_FLOAT_REGISTERS,
    floats::FloatLike,
    inputs::FloatSet,
    operation::{Benchmark, Operation},
};
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use hwlocality::Topology;
use rand::prelude::*;
#[cfg(feature = "simd")]
use std::simd::Simd;

// --- Benchmark configuration and steering ---

/// Maximum granularity of subnormal occurence probabilities
///
/// Higher is more precise, but the benchmark execution time in the default
/// configuration is multipled accordingly.
pub const MAX_SUBNORMAL_CONFIGURATIONS: usize = const {
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

/// Criterion benchmark entry point
pub fn criterion_benchmark(criterion: &mut Criterion) {
    // Find out how many bytes of data we can reliably fits in L1, L2, ... and
    // add a dataset size that only fits in RAM for completeness
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
        .collect::<Vec<_>>();

    // Define configuration that is shared by all benchmarks
    let config = &mut CommonConfiguration {
        rng: rand::thread_rng(),
        criterion,
        memory_input_sizes: &memory_input_sizes,
    };

    // Benchmark all selected data types
    //
    // We start with the types for which the impact of all observed effects is
    // expected to be the highest, namely the widest SIMD types, then we go down
    // in width until we reach scalar types.
    //
    // Leading zeros in the SIMD type names are a workaround for criterion's
    // poor benchmark name sorting logic. You will find a lot more of these
    // workarounds throughout this codebase...
    #[cfg(feature = "simd")]
    {
        // TODO: Expand pessimize's SIMD support in order to to cover more SIMD
        //       types from NEON, SVE, RISC-V vector extensions... The way this
        //       benchmark is currently designed, we can support any vector
        //       instruction set for which the following two conditions are met:
        //
        //       1. The portable Simd type from std implements a conversion to
        //          and from a matching architectural SIMD type.
        //       2. Rust's inline ASM, which we use as our optimization barrier,
        //          accepts SIMD register operands of that type.
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        benchmark_type::<Simd<f64, 2>>(config, "f64x02");
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(target_feature = "avx512f")]
            {
                benchmark_type::<Simd<f32, 16>>(config, "f32x16");
                benchmark_type::<Simd<f64, 8>>(config, "f64x08");
            }
            #[cfg(target_feature = "avx")]
            {
                benchmark_type::<Simd<f32, 8>>(config, "f32x08");
                benchmark_type::<Simd<f64, 4>>(config, "f64x04");
            }
            #[cfg(target_feature = "sse")]
            benchmark_type::<Simd<f32, 4>>(config, "f32x04");
            #[cfg(target_feature = "sse2")]
            benchmark_type::<Simd<f64, 2>>(config, "f64x02");
        }
    }
    benchmark_type::<f32>(config, "f32");
    benchmark_type::<f64>(config, "f64");
}

/// Configuration shared by all benchmarks
struct CommonConfiguration<'criterion, 'memory_input_sizes> {
    /// Random number generator
    rng: ThreadRng,

    /// Criterion benchmark harness
    criterion: &'criterion mut Criterion,

    /// Name and size in bytes of memory inputs
    memory_input_sizes: &'memory_input_sizes [(String, usize)],
}

/// Run all benchmarks for a given scalar/SIMD type
#[inline(never)] // Faster build + easier profiling
fn benchmark_type<T: FloatLike>(common_config: &mut CommonConfiguration, tname: &'static str) {
    // Set up memory inputs for this type
    let mut memory_inputs = common_config
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

    // Define the common configuration for this type
    let type_config = &mut TypeConfiguration {
        rng: common_config.rng.clone(),
        criterion: common_config.criterion,
        tname,
        memory_inputs: &mut memory_inputs,
    };

    // Benchmark each enabled operation
    #[cfg(feature = "bench_addsub")]
    benchmark_operation::<_, addsub::AddSub>(type_config);
    #[cfg(feature = "bench_sqrt_positive_addsub")]
    benchmark_operation::<_, sqrt_positive_addsub::SqrtPositiveAddSub>(type_config);
    #[cfg(feature = "bench_average")]
    benchmark_operation::<_, average::Average>(type_config);
    #[cfg(feature = "bench_mul_average")]
    benchmark_operation::<_, mul_average::MulAverage>(type_config);
    #[cfg(feature = "bench_fma_multiplier_average")]
    benchmark_operation::<_, fma_multiplier_average::FmaMultiplierAverage>(type_config);
    #[cfg(feature = "bench_fma_addend_average")]
    benchmark_operation::<_, fma_addend_average::FmaAddendAverage>(type_config);
    #[cfg(feature = "bench_fma_full_average")]
    benchmark_operation::<_, fma_full_average::FmaFullAverage>(type_config);
}

/// Configuration selected by [`benchmark_type()`]
struct TypeConfiguration<'criterion, 'memory_inputs, 'memory_input_name, T: FloatLike> {
    /// Random number generator
    rng: ThreadRng,

    /// Criterion benchmark harness
    criterion: &'criterion mut Criterion,

    /// Name of the type that is being benchmarked
    tname: &'static str,

    /// Preallocated buffers for the benchmark's memory inputs
    memory_inputs: &'memory_inputs mut [(&'memory_input_name str, Box<[T]>)],
}

/// Benchmark a certain data type, in a certain ILP configurations, using input
/// data from memory (CPU cache or RAM)
macro_rules! for_each_ilp {
    // Decide which ILP configurations we are going to instantiate...
    ( benchmark_memory_inputs $args:tt with {
        common_config: $common_config:tt,
        selected_ilp: $selected_ilp:expr,
    } ) => {
        for_each_ilp!(
            // Currently known CPU microarchitectures have at most 32 CPU
            // registers, so there is no point in instantiating ILP >32.
            benchmark_memory_inputs $args with {
                common_config: $common_config,
                selected_ilp: $selected_ilp,
                instantiated_ilps: [1, 2, 4, 8, 16, 32],
            }
        );
    };

    // ...then instantiate all these ILP configurations, pick the one currently
    // selected by the outer ILP loop in benchmark_operation(), and run it if
    // should_check_ilp says that we should do so.
    ( benchmark_memory_inputs( $input_storage:expr ) with {
        common_config: {
            rng: $rng:expr,
            float: $float:ty,
            operation: $operation:ty,
            group: $group:expr,
        },
        selected_ilp: $selected_ilp:expr,
        instantiated_ilps: [ $($instantiated_ilp:literal),* ],
    } ) => {
        // Check if the current ILP configuration should be benchmarked
        if should_check_ilp::<$operation, &mut [$float]>($selected_ilp) {
            // If so, find it in the set of instantiated ILP configurations...
            match $selected_ilp {
                // Instantiate all the ILP configurations
                $(
                    $instantiated_ilp => {
                        let benchmark = <$operation>::make_benchmark::<$instantiated_ilp>();
                        let config = BenchmarkConfiguration {
                            rng: $rng,
                            group: $group,
                            benchmark,
                        };
                        benchmark_memory_inputs( config, $input_storage );
                    }
                )*
                _ => unimplemented!("Asked to run with un-instantiated ILP {}", $selected_ilp),
            }
        }
    };
}

/// Benchmark a certain data type in a certain ILP configurations, using input
/// data from CPU registers.
///
/// This macro is a more complex variation of for_each_ilp!() above, so you may
/// want to study this one first before you try to figure out that one.
#[cfg(feature = "register_data_sources")]
macro_rules! for_each_inputregs_and_ilp {
    // Decide which INPUT_REGISTERS configurations we are going to try...
    ( benchmark_register_inputs() with {
        common_config: $common_config:tt,
        selected_ilp: $selected_ilp:expr,
    } ) => {
        for_each_inputregs_and_ilp!(
            // In addition to the registers that we use for inputs, we need at
            // least one register for accumulators. So in our current
            // power-of-two register allocation scheme, we can have at most
            // MIN_FLOAT_REGISTERS/2 register inputs.
            //
            // The currently highest known amount of architectural float
            // registers is 32, therefore is no point in generating
            // configurations with >=32 register inputs at this point in time.
            //
            // I am also not covering single-register inputs because many
            // benchmarks require at least two inputs to work as expected.
            benchmark_register_inputs() with {
                common_config: $common_config,
                selected_ilp: $selected_ilp,
                inputregs: [2, 4, 8, 16],
            }
        );
    };

    // ...then iterate over these INPUT_REGISTERS configurations, giving each
    // its own Criterion benchmark group, and then decide which ILP
    // configurations we are going to instantiate...
    ( benchmark_register_inputs() with {
        common_config: {
            rng: $rng:expr,
            float: $float:ty,
            operation: $operation:ty,
            criterion: $criterion:expr,
            group_name_prefix: $group_name_prefix:expr,
        },
        selected_ilp: $selected_ilp:expr,
        inputregs: [ $($inputregs:literal),* ],
    } ) => {
        // Iterate over all instantiated register input configurations
        $({
            // Set up a criterion group for this register input configuration
            let data_source_name = if $inputregs == 1 {
                // Leading zeros works around poor criterion bench name sorting
                "01register".to_string()
            } else {
                format!("{:02}registers", $inputregs)
            };
            let mut group = $criterion.benchmark_group(format!("{}/{data_source_name}", $group_name_prefix));

            // Dispatch to the selected ILP configuration
            for_each_inputregs_and_ilp!(
                // We need 1 register per accumulator and current hardware has
                // at most 32 float registers, so it does not make sense to
                // compile for accumulator ILP >= 32 at this point in time as we
                // would have no registers left for input.
                benchmark_register_inputs::<_, $inputregs>() with {
                    common_config: {
                        rng: $rng,
                        float: $float,
                        operation: $operation,
                        group: &mut group,
                    },
                    selected_ilp: $selected_ilp,
                    instantiated_ilps: [1, 2, 4, 8, 16],
                }
            );
        })*
    };

    // ...then instantiate all these ILP configurations, pick the one currently
    // selected by the outer ILP loop in benchmark_operation(), and decide if we
    // are going to run a benchmark with this degree of ILP or not...
    ( benchmark_register_inputs::< _, $inputregs:literal >() with {
        common_config: {
            rng: $rng:expr,
            float: $float:ty,
            operation: $operation:ty,
            group: $group:expr,
        },
        selected_ilp: $selected_ilp:expr,
        instantiated_ilps: [ $($instantiated_ilp:literal),* ],
    } ) => {
        if should_check_ilp::<$operation, [$float; $inputregs]>($selected_ilp) {
            match $selected_ilp {
                $(
                    $instantiated_ilp => {
                        let benchmark = <$operation>::make_benchmark::<$instantiated_ilp>();
                        let config = BenchmarkConfiguration {
                            rng: $rng,
                            group: $group,
                            benchmark,
                        };
                        benchmark_register_inputs::<_, $inputregs>(config);
                    }
                )*
                _ => unimplemented!("Asked to run with un-instantiated ILP {}", $selected_ilp),
            }
        }
    };
}

/// Benchmark a certain operation on data of a certain scalar/SIMD type
#[inline(never)] // Faster build + easier profiling
fn benchmark_operation<T: FloatLike, Op: Operation<T>>(type_config: &mut TypeConfiguration<T>) {
    // ...and for each supported degree of ILP...
    for ilp in (0..=MIN_FLOAT_REGISTERS.ilog2()).map(|ilp_pow2| 2usize.pow(ilp_pow2)) {
        // Name this (type, benchmark, ilp) triplet
        let ilp_name = if ilp == 1 {
            "chained".to_string()
        } else {
            // Leading zeros works around poor criterion bench name sorting
            format!("ilp{ilp:02}")
        };
        let group_name_prefix = format!("{}/{}/{ilp_name}", type_config.tname, Op::NAME);

        // Benchmark with register inputs, if configured to do so
        #[cfg(feature = "register_data_sources")]
        for_each_inputregs_and_ilp!(
            benchmark_register_inputs() with {
                common_config: {
                    rng: type_config.rng.clone(),
                    float: T,
                    operation: Op,
                    criterion: &mut type_config.criterion,
                    group_name_prefix: group_name_prefix,
                },
                selected_ilp: ilp,
            }
        );

        // Benchmark with all configured memory inputs
        for (data_source_name, input_storage) in type_config.memory_inputs.iter_mut() {
            // Set up a criterion group for this input configuration
            let mut group = type_config
                .criterion
                .benchmark_group(format!("{group_name_prefix}/{data_source_name}"));

            // Run the benchmarks at each supported ILP level
            for_each_ilp!(benchmark_memory_inputs(&mut *input_storage) with {
                common_config: {
                    rng: type_config.rng.clone(),
                    float: T,
                    operation: Op,
                    group: &mut group,
                },
                selected_ilp: ilp,
            });
        }
    }
}

/// Truth that we should benchmark a certain operation, with a certain degree of
/// instruction-level parallelism, on a certain input data source
fn should_check_ilp<Op, Inputs>(ilp: usize) -> bool
where
    Inputs: FloatSet,
    Op: Operation<Inputs::Element>,
{
    // First eliminate configurations that cannot fit in available CPU registers
    let non_accumulator_registers = if let Some(input_registers) = Inputs::NUM_REGISTER_INPUTS {
        input_registers + Op::AUX_REGISTERS_REGOP
    } else {
        Op::AUX_REGISTERS_MEMOP
    };
    if ilp + non_accumulator_registers > MIN_FLOAT_REGISTERS {
        return false;
    }

    // Unless the more_ilp_configurations feature is turned on, also
    // eliminate ILP configurations other than the minimum, maximum, and
    // half-maximum ILP for memory operands.
    if cfg!(feature = "more_ilp_configurations") {
        true
    } else {
        let max_ilp_memop = MIN_FLOAT_REGISTERS - Op::AUX_REGISTERS_MEMOP;
        // Round to lower power of two since we only benchmark at powers of two
        let optimal_ilp_memop = 1 << max_ilp_memop.ilog2();
        ilp == 1 || ilp == optimal_ilp_memop || ilp == optimal_ilp_memop / 2
    }
}

/// Configuration selected by [`benchmark_operation()`]
struct BenchmarkConfiguration<'criterion, 'group, B: Benchmark> {
    /// Random number generator
    rng: ThreadRng,

    /// Criterion benchmark group
    group: &'group mut BenchmarkGroup<'criterion, WallTime>,

    /// Benchmark
    benchmark: B,
}

/// Run a certain benchmark for a certain scalar/SIMD type, using a certain
/// degree of ILP and a certain number of register inputs.
#[cfg(feature = "register_data_sources")]
#[inline(never)] // Faster build + easier profiling
fn benchmark_register_inputs<B: Benchmark, const INPUT_REGISTERS: usize>(
    config: BenchmarkConfiguration<B>,
) {
    // Iterate over subnormal configurations
    let num_subnormal_configurations = INPUT_REGISTERS.min(MAX_SUBNORMAL_CONFIGURATIONS);
    for subnormal_share in 0..=num_subnormal_configurations {
        // Generate input data
        let num_subnormals = subnormal_share * INPUT_REGISTERS / num_subnormal_configurations;
        let mut inputs = [B::Float::default(); INPUT_REGISTERS];
        inputs.generate_positive(&mut *config.rng, num_subnormals);

        // Name this subnormal configuration
        let input_name = format!(
            "{num_subnormals:0num_digits$}in{INPUT_REGISTERS}",
            // Leading zeros works around poor criterion bench name sorting
            num_digits = INPUT_REGISTERS.ilog10() as usize + 1
        );

        // Run all the benchmarks on this input
        config
            .benchmark
            .run(config.group, inputs, input_name, &mut *config.rng);
    }
}

/// Run a certain benchmark for a certain scalar/SIMD type, using a certain
/// degree of ILP and memory inputs of a certain size.
#[inline(never)] // Faster build + easier profiling
fn benchmark_memory_inputs<B: Benchmark>(
    config: BenchmarkConfiguration<B>,
    mut input_storage: &mut [B::Float],
) {
    // Iterate over subnormal configurations
    for subnormal_probability in 0..=MAX_SUBNORMAL_CONFIGURATIONS {
        // Generate input data
        let subnormal_probability =
            subnormal_probability as f64 / MAX_SUBNORMAL_CONFIGURATIONS as f64;
        input_storage.generate_positive(
            &mut *config.rng,
            (subnormal_probability * input_storage.len() as f64).round() as usize,
        );

        // Name this subnormal configuration
        let input_name = format!(
            // Leading zeros works around poor criterion bench name sorting
            "{:0>5.1}%",
            subnormal_probability * 100.0,
        );

        // Run all the benchmarks on this input
        config.benchmark.run(
            config.group,
            &mut *input_storage,
            input_name,
            &mut *config.rng,
        );
    }
}

// --- Criterion boilerplate ---

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
