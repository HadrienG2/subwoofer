//! Benchmarking of individual operations on a certain floating-point type

use crate::inputs::{self, DataSourceConfiguration};
use common::{
    arch::MIN_FLOAT_REGISTERS, floats::FloatLike, inputs::FloatSet, operation::Operation,
};
use criterion::Criterion;
use rand::prelude::*;

/// Configuration selected by [`benchmark_type()`]
pub(crate) struct TypeConfiguration<'criterion, 'memory_inputs, 'memory_input_name, T: FloatLike> {
    /// Random number generator
    pub rng: ThreadRng,

    /// Criterion benchmark harness
    pub criterion: &'criterion mut Criterion,

    /// Name of the type that is being benchmarked
    pub t_name: &'static str,

    /// Preallocated buffers for the benchmark's memory inputs
    pub memory_inputs: &'memory_inputs mut [(&'memory_input_name str, Box<[T]>)],
}

/// Benchmark all enabled operations for a certain data type
pub(crate) fn benchmark_all<T: FloatLike>(mut config: TypeConfiguration<T>) {
    #[cfg(feature = "bench_addsub")]
    benchmark_operation::<_, addsub::AddSub>(&mut config);
    #[cfg(feature = "bench_sqrt_positive_addsub")]
    benchmark_operation::<_, sqrt_positive_addsub::SqrtPositiveAddSub>(&mut config);
    #[cfg(feature = "bench_average")]
    benchmark_operation::<_, average::Average>(&mut config);
    #[cfg(feature = "bench_mul_average")]
    benchmark_operation::<_, mul_average::MulAverage>(&mut config);
    #[cfg(feature = "bench_fma_multiplier_average")]
    benchmark_operation::<_, fma_multiplier_average::FmaMultiplierAverage>(&mut config);
    #[cfg(feature = "bench_fma_addend_average")]
    benchmark_operation::<_, fma_addend_average::FmaAddendAverage>(&mut config);
    #[cfg(feature = "bench_fma_full_average")]
    benchmark_operation::<_, fma_full_average::FmaFullAverage>(&mut config);
}

/// Benchmark a certain data type, in a certain ILP configurations, using input
/// data from memory (CPU cache or RAM)
macro_rules! for_each_ilp {
    // Decide which ILP configurations we are going to instantiate...
    ( inputs::benchmark_memory $args:tt with {
        common_config: $common_config:tt,
        selected_ilp: $selected_ilp:expr,
    } ) => {
        for_each_ilp!(
            // Currently known CPU microarchitectures have at most 32 CPU
            // registers, so there is no point in instantiating ILP >32.
            inputs::benchmark_memory $args with {
                common_config: $common_config,
                selected_ilp: $selected_ilp,
                instantiated_ilps: [1, 2, 4, 8, 16, 32],
            }
        );
    };

    // ...then instantiate all these ILP configurations, pick the one currently
    // selected by the outer ILP loop in benchmark_operation(), and run it if
    // should_check_ilp says that we should do so.
    ( inputs::benchmark_memory( $input_storage:expr ) with {
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
                        let config = DataSourceConfiguration {
                            rng: $rng,
                            group: $group,
                            benchmark,
                        };
                        inputs::benchmark_memory( config, $input_storage );
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
    ( inputs::benchmark_registers() with {
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
            inputs::benchmark_registers() with {
                common_config: $common_config,
                selected_ilp: $selected_ilp,
                inputregs: [2, 4, 8, 16],
            }
        );
    };

    // ...then iterate over these INPUT_REGISTERS configurations, giving each
    // its own Criterion benchmark group, and then decide which ILP
    // configurations we are going to instantiate...
    ( inputs::benchmark_registers() with {
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
                inputs::benchmark_registers::<_, $inputregs>() with {
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
    ( inputs::benchmark_registers::< _, $inputregs:literal >() with {
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
                        let config = DataSourceConfiguration {
                            rng: $rng,
                            group: $group,
                            benchmark,
                        };
                        inputs::benchmark_registers::<_, $inputregs>(config);
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
    // For each supported degree of instruction-level parallelism...
    for ilp in (0..=MIN_FLOAT_REGISTERS.ilog2()).map(|ilp_pow2| 2usize.pow(ilp_pow2)) {
        // Name this (type, benchmark, ilp) triplet
        let ilp_name = if ilp == 1 {
            "chained".to_string()
        } else {
            // Leading zeros works around poor criterion bench name sorting
            format!("ilp{ilp:02}")
        };
        let group_name_prefix = format!("{}/{}/{ilp_name}", type_config.t_name, Op::NAME);

        // Benchmark with register inputs, if configured to do so
        #[cfg(feature = "register_data_sources")]
        for_each_inputregs_and_ilp!(
            inputs::benchmark_registers() with {
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
            for_each_ilp!(inputs::benchmark_memory(&mut *input_storage) with {
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
