#![feature(portable_simd)]

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
use hwlocality::Topology;
use pessimize::Pessimize;
use rand::{distributions::Uniform, prelude::*};
use std::{
    num::NonZeroUsize,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    simd::{LaneCount, Simd, StdFloat, SupportedLaneCount},
    time::Instant,
};
use target_features::Architecture;

// --- Benchmark configuration and steering ---

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

/// Criterion benchmark entry point
pub fn criterion_benchmark(c: &mut Criterion) {
    // Collect the list of benchmarked operations
    let mut benchmark_names = Vec::new();
    if cfg!(feature = "bench_addsub") {
        benchmark_names.push("addsub")
    }
    if cfg!(feature = "bench_sqrt_positive_addsub") {
        benchmark_names.push("sqrt_positive_addsub")
    }
    if cfg!(feature = "bench_average") {
        benchmark_names.push("average");
    }
    if cfg!(feature = "bench_mul_average") {
        benchmark_names.push("mul_average")
    }
    if cfg!(feature = "bench_fma_multiplier_average") {
        benchmark_names.push("fma_multiplier_average");
    }
    if cfg!(feature = "bench_fma_addend_average") {
        benchmark_names.push("fma_addend_average")
    }
    if cfg!(feature = "bench_fma_full_average") {
        benchmark_names.push("fma_full_average");
    }

    // Find out how many bytes of data we can reliably fits in L1, L2, ... and
    // add a dataset size that only fits in RAM for completeness
    let cache_stats = Topology::new().unwrap().cpu_cache_stats().unwrap();
    let smallest_data_cache_sizes = cache_stats.smallest_data_cache_sizes();
    let max_size_to_fit = |cache_size: u64| cache_size / 2;
    let min_size_to_overflow = |cache_size: u64| cache_size * 8;
    let memory_input_sizes = if cfg!(feature = "more_memory_data_sources") {
        smallest_data_cache_sizes
            .iter()
            .copied()
            .map(max_size_to_fit)
            .chain(std::iter::once(min_size_to_overflow(
                *cache_stats.total_data_cache_sizes().last().unwrap(),
            )))
            .map(|size| usize::try_from(size).unwrap())
            .collect::<Vec<_>>()
    } else {
        vec![usize::try_from(max_size_to_fit(*smallest_data_cache_sizes.first().unwrap())).unwrap()]
    };

    // Then we will benchmark for each supported floating-point type
    let config = CommonConfiguration {
        benchmark_names: &benchmark_names,
        memory_input_sizes: &memory_input_sizes,
    };

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
        benchmark_type::<Simd<f64, 2>>(c, config, "f64x02");
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(target_feature = "avx512f")]
            {
                benchmark_type::<Simd<f32, 16>>(c, config, "f32x16");
                benchmark_type::<Simd<f64, 8>>(c, config, "f64x08");
            }
            #[cfg(target_feature = "avx")]
            {
                benchmark_type::<Simd<f32, 8>>(c, config, "f32x08");
                benchmark_type::<Simd<f64, 4>>(c, config, "f64x04");
            }
            #[cfg(target_feature = "sse")]
            benchmark_type::<Simd<f32, 4>>(c, config, "f32x04");
            #[cfg(target_feature = "sse2")]
            benchmark_type::<Simd<f64, 2>>(c, config, "f64x02");
        }
    }
    benchmark_type::<f32>(c, config, "f32");
    benchmark_type::<f64>(c, config, "f64");
}

/// Configuration that is common to all benchmarks
#[derive(Copy, Clone)]
struct CommonConfiguration<'a> {
    /// Operations that are being benchmarked
    benchmark_names: &'a [&'static str],

    /// Sizes (in bytes) of the memory inputs that benchmarks are fed with
    memory_input_sizes: &'a [usize],
}

/// Benchmark a certain data type in a certain ILP configurations, using input
/// data from memory (CPU cache or RAM)
macro_rules! for_each_ilp {
    // Decide which ILP configurations we are going to instantiate...
    ( $benchmark:ident ::< $t:ty > $args:tt with { selected_ilp: $selected_ilp:expr } ) => {
        for_each_ilp!(
            $benchmark::<$t> $args with { selected_ilp: $selected_ilp, instantiated_ilps: [1, 2, 4, 8, 16, 32] }
        );
    };

    // ...then generate all these ILP configurations, pick the one currently
    // selected by the outer ILP loop in benchmark_type(), and make sure that
    // this configuration can fit in CPU registers for at least one benchmark...
    ( $benchmark:ident ::< $t:ty > $args:tt with { selected_ilp: $selected_ilp:expr, instantiated_ilps: [ $($instantiated_ilp:literal),* ] } ) => {
        // On CPU architectures without memory operands, we need at least one
        // CPU register to load memory inputs even in the simplest benchmark.
        let input_operand_registers = if HAS_MEMORY_OPERANDS { 0 } else { 1 };
        match $selected_ilp {
            $(
                $instantiated_ilp => if $instantiated_ilp <= MIN_FLOAT_REGISTERS - input_operand_registers {
                    $benchmark::<$t, $instantiated_ilp> $args
                }
            )*
            _ => unimplemented!("Asked to run with unsupported ILP {}", $selected_ilp),
        }
    };
}

/// Benchmark a certain data type in a certain ILP configurations, using input
/// data from CPU registers.
///
/// This macro is a more complex variation of for_each_ilp!() above, so you may
/// want to study this one first before you try to figure out that one.
#[cfg(feature = "register_data_sources")]
macro_rules! for_each_registers_and_ilp {
    // Decide which INPUT_REGISTERS configurations we are going to try...
    ( $benchmark:ident ::< $t:ty > $args:tt with { criterion: $criterion:expr, type_config: $type_config:expr } ) => {
        for_each_registers_and_ilp!(
            // In addition to the registers that we use for inputs, we need at
            // least one register for accumulators.
            //
            // So in our current power-of-two register allocation scheme, we can
            // have at most MIN_FLOAT_REGISTERS/2 register inputs. The currently
            // highest known amount of architectural float registers is 32, so
            // there is no point in generating configurations with >=32 register
            // inputs at this point in time.
            //
            // I am also not covering single-register inputs because many
            // benchmarks require at least two inputs to work as expected.
            $benchmark::<$t> $args with { criterion: $criterion, type_config: $type_config, inputregs: [2, 4, 8, 16] }
        );
    };

    // ...then iterate over these INPUT_REGISTERS configurations, giving each
    // its own Criterion benchmark group, and then decide which ILP
    // configurations we are going to instantiate...
    ( $benchmark:ident ::< $t:ty > $args:tt with { criterion: $criterion:expr, type_config: $type_config:expr, inputregs: [ $($inputregs:literal),* ] } ) => {
        $({
            // Set up a criterion group for this input configuration
            let data_source_name = if $inputregs == 1 {
                // Leading zeros works around poor criterion bench name sorting
                "01register".to_string()
            } else {
                format!("{:02}registers", $inputregs)
            };
            let mut group = $criterion.benchmark_group(format!("{}/{data_source_name}", $type_config.group_name_prefix));

            // Dispatch to the proper ILP configurations
            for_each_registers_and_ilp!(
                // We need 1 register per accumulator and current hardware has
                // at most 32 float registers, so it does not make sense to
                // compile for ILP >= 32 at this point in time as we would have
                // no registers left for input. But we do need to compile in
                // ILP32 to avoid the unimplemented error below because ILP32 is
                // actually used for memory inputs.
                $benchmark::<$t, $inputregs> $args with { group: &mut group, selected_ilp: $type_config.ilp, instantiated_ilps: [1, 2, 4, 8, 16, 32] }
            );
        })*
    };

    // ...then generate all these ILP configurations, pick the one currently
    // selected by the outer ILP loop in benchmark_type(), and make sure that
    // this configuration can fit in CPU registers for at least one benchmark...
    ( $benchmark:ident ::< $t:ty, $inputregs:literal > $args:tt with { group: $group:expr, selected_ilp: $selected_ilp:expr, instantiated_ilps: [ $($instantiated_ilp:literal),* ] } ) => {
        match $selected_ilp {
            $(
                // Need at least $inputregs registers for registers for register
                // inputs and $ilp registers for accumulators
                $instantiated_ilp => if ($instantiated_ilp + $inputregs) <= MIN_FLOAT_REGISTERS {
                    for_each_registers_and_ilp!(
                        $benchmark::<$t, $inputregs, $instantiated_ilp> $args with { group: $group }
                    );
                }
            )*
            _ => unimplemented!("Asked to run with unsupported ILP {}", $selected_ilp),
        }
    };

    // ...finally, add the criterion group to the list of benchmark arguments
    // and call the inner benchmark worker.
    ( $benchmark:ident ::< $t:ty, $inputregs:literal, $ilp:literal >( $($arg:expr),* ) with { group: $group:expr } ) => {
        $benchmark::<$t, $inputregs, $ilp>( $($arg,)* $group );
    };
}

/// Run all benchmarks for a given scalar/SIMD type
#[inline(never)] // Faster build + easier profiling
fn benchmark_type<T: FloatLike>(
    c: &mut Criterion,
    common_config: CommonConfiguration,
    tname: &str,
) {
    // For each benchmarked operation...
    for benchmark_name in common_config.benchmark_names {
        // ...and for each supported degree of ILP...
        for ilp in (0..=MIN_FLOAT_REGISTERS.ilog2()).map(|ilp_pow2| 2usize.pow(ilp_pow2)) {
            // Name this (type, benchmark, ilp) triplet
            let ilp_name = if ilp == 1 {
                "chained".to_string()
            } else {
                // Leading zeros works around poor criterion bench name sorting
                format!("ilp{ilp:02}")
            };
            let group_name_prefix = format!("{tname}/{benchmark_name}/{ilp_name}");

            // Benchmark with input data that fits in CPU registers
            #[cfg(feature = "register_data_sources")]
            {
                let type_config = TypeConfiguration {
                    ilp,
                    group_name_prefix: &group_name_prefix,
                };
                for_each_registers_and_ilp!(benchmark_register_inputs::<T>(benchmark_name) with { criterion: c, type_config: &type_config });
            }

            // Benchmark with input data that fits in L1, L2, ... all the way to RAM
            for (idx, &input_size) in common_config.memory_input_sizes.iter().enumerate() {
                // Allocate storage for input data
                let num_elems = input_size / std::mem::size_of::<T>();
                let mut input_storage = vec![T::default(); num_elems];

                // Set up a criterion group for this input configuration
                let data_source_name = if cfg!(feature = "more_memory_data_sources") {
                    if idx < common_config.memory_input_sizes.len() - 1 {
                        format!("L{}cache", idx + 1)
                    } else {
                        "RAM".to_string()
                    }
                } else {
                    assert_eq!(common_config.memory_input_sizes.len(), 1);
                    "L1cache".to_string()
                };
                let mut group =
                    c.benchmark_group(format!("{group_name_prefix}/{data_source_name}"));

                // Run the benchmarks at each supported ILP level
                for_each_ilp!(benchmark_memory_inputs::<T>(benchmark_name, &mut group, &mut input_storage) with { selected_ilp: ilp });
            }
        }
    }
}

/// Configuration selected by the outer double loop of `benchmark_type()`
#[cfg(feature = "register_data_sources")]
struct TypeConfiguration<'group_name_prefix> {
    /// Selected degree of ILP
    ilp: usize,

    /// Common prefix for the names of all inner criterion benchmark groups
    group_name_prefix: &'group_name_prefix str,
}

/// Run a certain benchmark for a certain scalar/SIMD type, using a certain
/// degree of ILP and a certain number of register inputs.
#[cfg(feature = "register_data_sources")]
#[inline(never)] // Faster build + easier profiling
fn benchmark_register_inputs<T: FloatLike, const INPUT_REGISTERS: usize, const ILP: usize>(
    benchmark_name: &'static str,
    group: &mut BenchmarkGroup<WallTime>,
) {
    // Iterate over subnormal configurations
    let num_subnormal_configurations = INPUT_REGISTERS.min(MAX_SUBNORMAL_CONFIGURATIONS);
    for subnormal_share in 0..=num_subnormal_configurations {
        // Generate input data
        let num_subnormals = subnormal_share * INPUT_REGISTERS / num_subnormal_configurations;
        let inputs = T::generate_positive_input_array::<INPUT_REGISTERS>(num_subnormals);

        // Name this subnormal configuration
        let input_name = format!(
            "{num_subnormals:0num_digits$}in{INPUT_REGISTERS}",
            // Leading zeros works around poor criterion bench name sorting
            num_digits = INPUT_REGISTERS.ilog10() as usize + 1
        );

        // Run all the benchmarks on this input
        benchmark_input_set::<_, _, ILP>(benchmark_name, group, inputs, input_name);
    }
}

/// Run a certain benchmark for a certain scalar/SIMD type, using a certain
/// degree of ILP and memory inputs of a certain size.
#[inline(never)] // Faster build + easier profiling
fn benchmark_memory_inputs<T: FloatLike, const ILP: usize>(
    benchmark_name: &'static str,
    group: &mut BenchmarkGroup<WallTime>,
    input_storage: &mut [T],
) {
    // Iterate over subnormal configurations
    for subnormal_probability in 0..=MAX_SUBNORMAL_CONFIGURATIONS {
        // Generate input data
        let subnormal_probability =
            subnormal_probability as f64 / MAX_SUBNORMAL_CONFIGURATIONS as f64;
        T::generate_positive_inputs(
            input_storage,
            (subnormal_probability * input_storage.len() as f64).round() as usize,
        );

        // Name this subnormal configuration
        let input_name = format!(
            // Leading zeros works around poor criterion bench name sorting
            "{:0>5.1}%",
            subnormal_probability * 100.0,
        );

        // Run all the benchmarks on this input
        benchmark_input_set::<_, _, ILP>(benchmark_name, group, &mut *input_storage, input_name);
    }
}

/// Run a certain benchmark for a certain scalar/SIMD type, using a certain
/// degree of ILP and a certain input data configuration.
#[inline(never)] // Faster build + easier profiling
fn benchmark_input_set<T: FloatLike, TSet: FloatSet<T>, const ILP: usize>(
    benchmark_name: &'static str,
    group: &mut BenchmarkGroup<WallTime>,
    mut inputs: TSet,
    input_name: String,
) {
    // Generate accumulator initial values and averaging targets
    let mut rng = rand::thread_rng();
    let normal_sampler = T::normal_sampler();
    // For additive random walks, a large initial accumulator value maximally
    // protects against descending into subnormal range as the accumulator value
    // gets smaller, and a small initial accumulator value maximally protects
    // against roundoff error as the accumulator value gets bigger. Both could
    // theoretically affect arithmetic perf on hypothetical Sufficiently Smart
    // Hardware from the future, so we protect against both equally.
    let additive_accumulator_init = std::array::from_fn::<_, ILP, _>(|_| {
        T::splat(2.0f32.powi((T::MANTISSA_DIGITS / 2) as i32)) * normal_sampler(&mut rng)
    });
    // For multiplicative random walks, accumulators close to 1 maximally
    // protect against exponent overflow and underflow.
    let averaging_accumulator_init = std::array::from_fn::<_, ILP, _>(|_| normal_sampler(&mut rng));
    let average_target = normal_sampler(&mut rng);

    // Throughput in operations/second is equal to throughput in data
    // inputs/second for almost all benchmarks, except for fma_full which
    // ingests two inputs per operation.
    let num_inputs = inputs.as_mut().len();
    let num_operation_inputs = if TSet::IS_REUSED {
        num_inputs * ILP // Each input is fed to each ILP stream
    } else {
        num_inputs // Each ILP stream gets its own substream of the input data
    } as u64;
    group.throughput(Throughput::Elements(num_operation_inputs));

    // Check if we should benchmark a certain ILP configuration, knowing how
    // much registers besides accumulators and register inputs are consumed when
    // operating from register and memory inputs.
    let should_check_ilp = |aux_registers_regop: usize, aux_registers_memop: usize| {
        // First eliminate configurations that would spill to the stack
        let non_accumulator_registers = if let Some(input_registers) = TSet::NUM_REGISTER_INPUTS {
            input_registers + aux_registers_regop
        } else {
            aux_registers_memop
        };
        if ILP + non_accumulator_registers > MIN_FLOAT_REGISTERS {
            return false;
        }

        // Unless the more_ilp_configurations feature is turned on, also
        // eliminate ILP configurations other than the minimum, maximum, and
        // half-maximum ILP for memory operands.
        if cfg!(feature = "more_ilp_configurations") {
            true
        } else {
            let max_ilp_memop = MIN_FLOAT_REGISTERS - aux_registers_memop;
            // Round to lower power of two since we only benchmark at powers of two
            let optimal_ilp_memop = 1 << max_ilp_memop.ilog2();
            ILP == 1 || ILP == optimal_ilp_memop || ILP == optimal_ilp_memop / 2
        }
    };

    // Most benchmarks take one input from registers or memory and use it
    // directly as an operand to an arithmetic operation.
    //
    // If the input comes from memory, the number of available registers depends
    // on whether the CPU ISA supports memory operands. If it doesn't, then an
    // extra temporary register must be used for a memory load before the
    // arithmetic operation can take place.
    let aux_registers_direct_memop = (!HAS_MEMORY_OPERANDS) as usize;

    // Run the benchmark currently selected by benchmark_type's outer loop
    'select_benchmark: {
        // Benchmark addition and subtraction
        #[cfg(feature = "bench_addsub")]
        if benchmark_name == "addsub" {
            if should_check_ilp(0, aux_registers_direct_memop) {
                run_benchmark(group, inputs, input_name, additive_accumulator_init, AddSub);
            }
            break 'select_benchmark;
        }

        // Benchmark square root of positive numbers, followed by add/sub cycle
        //
        // We always need one extra register to hold the square root before
        // adding or subtracting it from the accumulator. But if the
        // architecture doesn't support memory operands, we can reuse the one
        // that was used for the memory load.
        #[cfg(feature = "bench_sqrt_positive_addsub")]
        if benchmark_name == "sqrt_positive_addsub" {
            if should_check_ilp(1, 1) {
                run_benchmark(
                    group,
                    inputs,
                    input_name,
                    additive_accumulator_init,
                    SqrtPositiveAddSub,
                );
            }
            break 'select_benchmark;
        }

        // For multiplicative benchmarks, we're going to need to an extra
        // averaging operation, otherwise once we've multiplied by a subnormal
        // the accumulator stays in subnormal range forever, which get in the
        // way of our goal of studying the effect of various subnormals
        // occurence frequencies in the input stream.
        //
        // Two registers are used by the averaging (target + halving weight),
        // restricting available ILP for this benchmark.
        let aux_registers_average = 2;
        {
            if should_check_ilp(
                aux_registers_average,
                aux_registers_average + aux_registers_direct_memop,
            ) {
                // First benchmark the averaging in isolation
                #[cfg(feature = "bench_average")]
                if benchmark_name == "average" {
                    run_benchmark(
                        group,
                        inputs,
                        input_name,
                        averaging_accumulator_init,
                        Average,
                    );
                    break 'select_benchmark;
                }

                // Benchmark multiplication -> averaging
                #[cfg(feature = "bench_mul_average")]
                if benchmark_name == "mul_average" {
                    run_benchmark(
                        group,
                        inputs,
                        input_name,
                        averaging_accumulator_init,
                        MulAverage {
                            target: average_target,
                        },
                    );
                    break 'select_benchmark;
                }

                // Benchmark fma with possibly subnormal multiplier -> averaging
                #[cfg(feature = "bench_fma_multiplier_average")]
                if benchmark_name == "fma_multiplier_average" {
                    run_benchmark(
                        group,
                        inputs,
                        input_name,
                        averaging_accumulator_init,
                        FmaMultiplierAverage {
                            target: average_target,
                        },
                    );
                    break 'select_benchmark;
                }

                // Benchmark fma with possibly subnormal addend -> averaging
                #[cfg(feature = "bench_fma_addend_average")]
                if benchmark_name == "fma_addend_average" {
                    run_benchmark(
                        group,
                        inputs,
                        input_name,
                        averaging_accumulator_init,
                        FmaAddendAverage {
                            target: average_target,
                        },
                    );
                    break 'select_benchmark;
                }
            }
        }

        // Benchmark fma with possibly subnormal multiplier _and_ addend,
        // followed by averaging.
        //
        // This benchmark suffers from more register pressure than other
        // averaging-based benchmarks because there is one more operand to load
        // from memory before the FMA, even on CPU architectures with memory
        // operands (FMA can only accept 1 memory operand).
        #[cfg(feature = "bench_fma_full_average")]
        if benchmark_name == "fma_full_average" {
            let aux_registers_except_op1 = aux_registers_average + 1;
            if should_check_ilp(
                aux_registers_except_op1,
                aux_registers_except_op1 + aux_registers_direct_memop,
            ) {
                // Benchmark fma then averaging with possibly subnormal operands
                group.throughput(Throughput::Elements(num_operation_inputs / 2));
                run_benchmark(
                    group,
                    inputs,
                    input_name,
                    averaging_accumulator_init,
                    FmaFullAverage {
                        target: average_target,
                    },
                );
            }
            break 'select_benchmark;
        }
    }
}

/// Run a criterion benchmark of the specified computation
#[inline(never)] // Faster build + easier profiling
fn run_benchmark<'inputs, T: FloatLike, Inputs: FloatSet<T> + 'inputs, const ILP: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    mut inputs: Inputs,
    input_name: String,
    accumulator_init: [T; ILP],
    iteration: impl BenchmarkIteration<T, Inputs, ILP>,
) {
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
            // - Accumulators and register inputs are resident in CPU registers
            //   before the timer starts, eliminating possible bias for small
            //   numbers of iterations vs large numbers of iterations.
            let mut accumulators = <[T; ILP] as FloatSequence<T>>::hide(accumulator_init);
            let mut inputs = inputs.make_sequence();

            // Timed region, this is the danger zone where inlining and compiler
            // optimizations must be reviewed very carefully.
            let start = Instant::now();
            for _ in 0..iters {
                let (next_accs, next_inputs) = iteration.iterate(accumulators, inputs);
                accumulators = next_accs;
                inputs = next_inputs;
            }
            let elapsed = start.elapsed();

            // Tell the compiler that the accumulated results are used so it
            // doesn't delete the computation
            for acc in accumulators {
                pessimize::consume::<T>(acc);
            }
            elapsed
        });
    });
}

/// Workload that is being benchmarked by the inner loop of [`run_benchmark()`]
///
/// Takes a set of accumulators and a set of input data to be integrated into
/// them. Returns the updated accumulators and the original input data, but this
/// latter fact may or may not be hidden from the compiler's optimizer using a
/// [`pessimize::hide()`] optimization barrier.
///
/// Such an optimization barrier needs to be applied in benchmarks where the
/// compiler can perform unwanted optimizations (like hoisting one of the
/// operations whose performance we are trying to benchmark out of the inner
/// loop) if it knows that the inputs are always the same. It should not be used
/// when that is not the case as `pessimize::hide()` can cause harmful
/// side-effects like preventing the use of ASM memory operands.
///
/// A `T::hide_accumulator()` optimization barrier also needs to be applied to
/// each intermediary result before storing it back into the accumulator array.
/// That's because if the underlying data flow is fully exposed to LLVM, it will
/// figure out that we are doing similar things to each accumulator in an array,
/// using data from an identically sized slice of input data, and then it is
/// likely to try to be helpful by vectorizing the resulting array computation.
/// Unfortunately, this optimization is not wanted here because we want to
/// separately measure the overhead of subnormal numbers on scalar and SIMD
/// computations, and we cannot do that when our scalar computations are
/// helpfully turned into more SIMD computations by the compiler...
///
/// Finally, we ask that iteration callbacks be `Copy` as a hint that they
/// shouldn't capture any outer mutable data via `&mut X`. That's because at the
/// time of writing, the `pessimize::hide()` optimization barrier that we use
/// has the unfortunate side-effect of making rustc/LLVM spill the target of
/// there mutable references to memory and load them back from memory. This is
/// not appropriate when the data that we're using is supposed to stay resident
/// in CPU registers. To avoid this, all state mutation operations should be
/// carried out using by-value `Fn(T) -> T` style APIs.
trait BenchmarkIteration<T: FloatLike, Inputs: FloatSet<T>, const ILP: usize>: Copy {
    fn iterate(
        self,
        accs: [T; ILP],
        inputs: Inputs::Sequence<'_>,
    ) -> ([T; ILP], Inputs::Sequence<'_>);
}

// --- Actual benchmarks ---

#[cfg(feature = "bench_addsub")]
mod addsub {
    use super::*;

    /// Benchmark addition and subtraction
    ///
    /// This is just a random additive walk of ~unity or subnormal step, so given a
    /// high enough starting point, an initially normal accumulator should stay in
    /// the normal range forever.
    ///
    /// This benchmark directly integrates each input into the accumulator, hence...
    ///
    /// - On architectures with memory operands, all CPU registers can always be
    ///   used for register inputs and accumulators.
    /// - On architectures without memory operands...
    ///   * When operating from register inputs, all CPU registers that are not used
    ///     as inputs can be used as accumulators.
    ///   * When operating from memory, one CPU register must be reserved for memory
    ///     loads. The remaining CPU registers can be used as accumulators.
    #[derive(Clone, Copy)]
    pub(super) struct AddSub;
    //
    impl<T: FloatLike, Inputs: FloatSet<T>, const ILP: usize> BenchmarkIteration<T, Inputs, ILP>
        for AddSub
    {
        #[inline]
        fn iterate(
            self,
            accumulators: [T; ILP],
            inputs: Inputs::Sequence<'_>,
        ) -> ([T; ILP], Inputs::Sequence<'_>) {
            // No need for input hiding here, the compiler cannot do anything dangerous
            // with the knowledge that inputs are always the same in this benchmark.
            iter_halves::<_, _, ILP, false>(
                accumulators,
                inputs,
                |acc, elem| acc + elem,
                |acc, elem| acc - elem,
            )
        }
    }
}
#[cfg(feature = "bench_addsub")]
use addsub::AddSub;

#[cfg(feature = "bench_sqrt_positive_addsub")]
mod sqrt_positive_addsub {
    use super::*;

    /// Benchmark square root of positive numbers, followed by add/sub cycle
    ///
    /// Square roots of negative numbers may or may not be emulated in software.
    /// They are thus not a good candidate for CPU microbenchmarking.
    ///
    /// Square roots must be stored in one temporary register before add/sub,
    /// therefore one CPU register must always be reserved for square root
    /// temporaries, in addition to the registers used for inputs and accumulators.
    #[derive(Clone, Copy)]
    pub(super) struct SqrtPositiveAddSub;
    //
    impl<T: FloatLike, Inputs: FloatSet<T>, const ILP: usize> BenchmarkIteration<T, Inputs, ILP>
        for SqrtPositiveAddSub
    {
        #[inline]
        fn iterate(
            self,
            accumulators: [T; ILP],
            inputs: Inputs::Sequence<'_>,
        ) -> ([T; ILP], Inputs::Sequence<'_>) {
            let low_iter = |acc, elem: T| acc + elem.sqrt();
            let high_iter = |acc, elem: T| acc - elem.sqrt();
            if Inputs::IS_REUSED {
                // Need to hide reused register inputs, so that the compiler doesn't
                // abusively factor out the redundant square root computations and reuse
                // their result for all accumulators (in fact it would even be allowed
                // to reuse them for the entire outer iters loop in run_benchmark).
                iter_halves::<_, _, ILP, true>(accumulators, inputs, low_iter, high_iter)
            } else {
                // Memory inputs do not need to be hidden because each accumulator gets
                // its own input substream (preventing square root reuse during the
                // inner loop over accumulators) and current LLVM is not crazy enough to
                // precompute square roots for a whole arbitrarily large
                // dynamically-sized batch of input data.
                assert!(Inputs::NUM_REGISTER_INPUTS.is_none());
                iter_halves::<_, _, ILP, false>(accumulators, inputs, low_iter, high_iter)
            }
        }
    }
}
#[cfg(feature = "bench_sqrt_positive_addsub")]
use sqrt_positive_addsub::SqrtPositiveAddSub;

#[cfg(feature = "bench_average")]
mod average {
    use super::*;

    /// Benchmark rolling average with some data inputs.
    ///
    /// For multiplicative benchmarks, we're going to need to an extra averaging
    /// operation, otherwise once we've multiplied by a subnormal we'll stay in
    /// subnormal range forever.
    ///
    /// It cannot be just an addition, because otherwise if we have no subnormal
    /// input we get unbounded growth, which is also a problem.
    ///
    /// This benchmark measures the overhead of averaging with an in-register
    /// input in isolation, so that it can be subtracted from the overhead of X
    /// + averaging (with due respect paid to the existence of superscalar
    /// execution).
    ///
    /// At least one CPU register must be reserved to the averaging weight.
    /// Then...
    ///
    /// - If the input comes from registers, no further CPU register needs to
    ///   reserved because the initial accumulator register can be reused for
    ///   (acc + elem), and then the product of that by the averaging weight.
    /// - If the input comes from memory, then we need one extra CPU register
    ///   for the input memory load on architectures without memory operands,
    ///   before it can be summed with the accumulator. This is not necessary on
    ///   architectures with memory operands.
    #[derive(Clone, Copy)]
    pub(super) struct Average;
    //
    impl<T: FloatLike, Inputs: FloatSet<T>, const ILP: usize> BenchmarkIteration<T, Inputs, ILP>
        for Average
    {
        #[inline]
        fn iterate(
            self,
            accumulators: [T; ILP],
            inputs: Inputs::Sequence<'_>,
        ) -> ([T; ILP], Inputs::Sequence<'_>) {
            iter_full(accumulators, inputs, |acc, elem| {
                (acc + elem) * T::splat(0.5)
            })
        }
    }
}
#[cfg(feature = "bench_average")]
use average::Average;

#[cfg(feature = "bench_mul_average")]
mod mul_average {
    use super::*;

    /// Benchmark multiplication followed by averaging
    ///
    /// At least two CPU registers must be reserved to the averaging target and
    /// weight. Then...
    ///
    /// - If the input comes from registers, no further CPU register needs to
    ///   reserved because the initial accumulator register can be reused for (acc *
    ///   elem), then for sum with target, then for product by averaging weight.
    /// - If the input comes from memory, then we need one extra CPU register for
    ///   the input memory load on architectures without memory operands, before it
    ///   can be multiplied by the accumulator and stored into the former
    ///   accumulator register. No such extra register is needed on architectures
    ///   with memory operands.
    ///
    /// This is also true of all of the following `_average` computations that take
    /// a single input from the input data set.
    #[derive(Clone, Copy)]
    pub(super) struct MulAverage<T: FloatLike> {
        pub target: T,
    }
    //
    impl<T: FloatLike, Inputs: FloatSet<T>, const ILP: usize> BenchmarkIteration<T, Inputs, ILP>
        for MulAverage<T>
    {
        #[inline]
        fn iterate(
            self,
            accumulators: [T; ILP],
            inputs: Inputs::Sequence<'_>,
        ) -> ([T; ILP], Inputs::Sequence<'_>) {
            iter_full(accumulators, inputs, move |acc, elem| {
                (acc * elem + self.target) * T::splat(0.5)
            })
        }
    }
}
#[cfg(feature = "bench_mul_average")]
use mul_average::MulAverage;

#[cfg(feature = "bench_fma_multiplier_average")]
mod fma_multiplier_average {
    use super::*;

    /// Benchmark FMA with a possibly subnormal multiplier, followed by
    /// averaging
    #[derive(Clone, Copy)]
    pub(super) struct FmaMultiplierAverage<T: FloatLike> {
        pub target: T,
    }
    //
    impl<T: FloatLike, Inputs: FloatSet<T>, const ILP: usize> BenchmarkIteration<T, Inputs, ILP>
        for FmaMultiplierAverage<T>
    {
        #[inline]
        fn iterate(
            self,
            accumulators: [T; ILP],
            inputs: Inputs::Sequence<'_>,
        ) -> ([T; ILP], Inputs::Sequence<'_>) {
            let halve_weight = T::splat(0.5);
            iter_full(accumulators, inputs, move |acc, elem| {
                (acc.mul_add(elem, halve_weight) + self.target) * halve_weight
            })
        }
    }
}
#[cfg(feature = "bench_fma_multiplier_average")]
use fma_multiplier_average::FmaMultiplierAverage;

#[cfg(feature = "bench_fma_addend_average")]
mod fma_addend_average {
    use super::*;

    /// Benchmark FMA with a possibly subnormal addend, followed by averaging
    #[derive(Clone, Copy)]
    pub(super) struct FmaAddendAverage<T: FloatLike> {
        pub target: T,
    }
    //
    impl<T: FloatLike, Inputs: FloatSet<T>, const ILP: usize> BenchmarkIteration<T, Inputs, ILP>
        for FmaAddendAverage<T>
    {
        #[inline]
        fn iterate(
            self,
            accumulators: [T; ILP],
            inputs: Inputs::Sequence<'_>,
        ) -> ([T; ILP], Inputs::Sequence<'_>) {
            let halve_weight = T::splat(0.5);
            iter_full(accumulators, inputs, move |acc, elem| {
                (acc.mul_add(halve_weight, elem) + self.target) * halve_weight
            })
        }
    }
}
#[cfg(feature = "bench_fma_addend_average")]
use fma_addend_average::FmaAddendAverage;

#[cfg(feature = "bench_fma_full_average")]
mod fma_full_average {
    use super::*;

    /// Benchmark FMA with possibly subnormal inputs, followed by averaging
    ///
    /// In addition to the CPU registers used for register inputs and accumulators,
    /// this benchmark consumes...
    ///
    /// - A minimum or two CPU registers for the averaging target and weight
    /// - For memory inputs...
    ///   * One extra CPU register for at least one of the FMA memory operands,
    ///     because even on architecturs with memory operands, FMA can only take at
    ///     most one such operand.
    ///   * One extra CPU register for the other FMA memory operands on
    ///     architectures without memory operands.
    #[derive(Clone, Copy)]
    pub(super) struct FmaFullAverage<T: FloatLike> {
        pub target: T,
    }
    //
    impl<T: FloatLike, Inputs: FloatSet<T>, const ILP: usize> BenchmarkIteration<T, Inputs, ILP>
        for FmaFullAverage<T>
    {
        #[inline]
        fn iterate(
            self,
            mut accumulators: [T; ILP],
            inputs: Inputs::Sequence<'_>,
        ) -> ([T; ILP], Inputs::Sequence<'_>) {
            let iter = move |acc: T, factor, addend| {
                (acc.mul_add(factor, addend) + self.target) * T::splat(0.5)
            };
            let inputs_slice = inputs.as_ref();
            let (factor_inputs, addend_inputs) = inputs_slice.split_at(inputs_slice.len() / 2);
            if Inputs::IS_REUSED {
                assert_eq!(factor_inputs.len(), addend_inputs.len());
                for (&factor, &addend) in factor_inputs.iter().zip(addend_inputs) {
                    for acc in accumulators.iter_mut() {
                        *acc = iter(*acc, factor, addend);
                    }
                    accumulators = hide_accumulators(accumulators);
                }
            } else {
                let factor_chunks = factor_inputs.chunks_exact(ILP);
                let addend_chunks = addend_inputs.chunks_exact(ILP);
                let factor_remainder = factor_chunks.remainder();
                let addend_remainder = addend_chunks.remainder();
                for (factor_chunk, addend_chunk) in factor_chunks.zip(addend_chunks) {
                    for ((&factor, &addend), acc) in factor_chunk
                        .iter()
                        .zip(addend_chunk)
                        .zip(accumulators.iter_mut())
                    {
                        *acc = iter(*acc, factor, addend);
                    }
                    accumulators = hide_accumulators(accumulators);
                }
                for ((&factor, &addend), acc) in factor_remainder
                    .iter()
                    .zip(addend_remainder)
                    .zip(accumulators.iter_mut())
                {
                    *acc = iter(*acc, factor, addend);
                }
            }
            (accumulators, inputs)
        }
    }
}
#[cfg(feature = "bench_fma_full_average")]
use fma_full_average::FmaFullAverage;

/// Benchmark skeleton that processes the full input homogeneously
#[inline]
fn iter_full<T: FloatLike, Inputs: FloatSequence<T>, const ILP: usize>(
    mut accumulators: [T; ILP],
    inputs: Inputs,
    mut iter: impl Copy + FnMut(T, T) -> T,
) -> ([T; ILP], Inputs) {
    let inputs_slice = inputs.as_ref();
    if Inputs::IS_REUSED {
        for &elem in inputs_slice {
            for acc in accumulators.iter_mut() {
                *acc = iter(*acc, elem);
            }
            accumulators = hide_accumulators(accumulators);
        }
    } else {
        let chunks = inputs_slice.chunks_exact(ILP);
        let remainder = chunks.remainder();
        for chunk in chunks {
            for (&elem, acc) in chunk.iter().zip(accumulators.iter_mut()) {
                *acc = iter(*acc, elem);
            }
            accumulators = hide_accumulators(accumulators);
        }
        for (&elem, acc) in remainder.iter().zip(accumulators.iter_mut()) {
            *acc = iter(*acc, elem);
        }
    }
    (accumulators, inputs)
}

/// Benchmark skeleton that treats each half of the input differently
#[inline]
fn iter_halves<
    T: FloatLike,
    Inputs: FloatSequence<T>,
    const ILP: usize,
    const HIDE_INPUTS: bool,
>(
    mut accumulators: [T; ILP],
    mut inputs: Inputs,
    mut low_iter: impl Copy + FnMut(T, T) -> T,
    mut high_iter: impl Copy + FnMut(T, T) -> T,
) -> ([T; ILP], Inputs) {
    if Inputs::IS_REUSED {
        if HIDE_INPUTS {
            // When we need to hide inputs, we flip the order of iteration over
            // inputs and accumulators so that each set of inputs is fully
            // processed before we switch accumulators.
            //
            // This ensures that the optimization barrier is maximally allowed
            // to reuse the same registers for the original and hidden inputs,
            // at the expense of making it harder for the CPU to extract ILP if
            // the backend doesn't reorder instructions because the CPU frontend
            // must now dive through N mentions of the same accumulator before
            // reaching mentions of another accumulator (no unroll & jam).
            for acc in accumulators.iter_mut() {
                let inputs_slice = inputs.as_ref();
                let (low_inputs, high_inputs) = inputs_slice.split_at(inputs_slice.len() / 2);
                assert_eq!(low_inputs.len(), high_inputs.len());
                for (&low_elem, &high_elem) in low_inputs.iter().zip(high_inputs) {
                    *acc = low_iter(*acc, low_elem);
                    *acc = high_iter(*acc, high_elem);
                }
                // Autovectorization barrier must target one accumulator at a
                // time here due to the input/accumulator loop transpose.
                *acc = pessimize::hide::<T>(*acc);
                inputs = <Inputs as FloatSequence<T>>::hide(inputs);
            }
        } else {
            // Otherwise, we just do the same as usual, but with input reuse
            let inputs_slice = inputs.as_ref();
            let (low_inputs, high_inputs) = inputs_slice.split_at(inputs_slice.len() / 2);
            assert_eq!(low_inputs.len(), high_inputs.len());
            for (&low_elem, &high_elem) in low_inputs.iter().zip(high_inputs) {
                for acc in accumulators.iter_mut() {
                    *acc = low_iter(*acc, low_elem);
                    *acc = high_iter(*acc, high_elem);
                }
                accumulators = hide_accumulators(accumulators);
            }
        }
    } else {
        let inputs_slice = inputs.as_ref();
        let (low_inputs, high_inputs) = inputs_slice.split_at(inputs_slice.len() / 2);
        let low_chunks = low_inputs.chunks_exact(ILP);
        let high_chunks = high_inputs.chunks_exact(ILP);
        let low_remainder = low_chunks.remainder();
        let high_remainder = high_chunks.remainder();
        for (low_chunk, high_chunk) in low_chunks.zip(high_chunks) {
            for ((&low_elem, &high_elem), acc) in low_chunk
                .iter()
                .zip(high_chunk.iter())
                .zip(accumulators.iter_mut())
            {
                *acc = low_iter(*acc, low_elem);
                *acc = high_iter(*acc, high_elem);
            }
            accumulators = hide_accumulators(accumulators);
        }
        for (&low_elem, acc) in low_remainder.iter().zip(accumulators.iter_mut()) {
            *acc = low_iter(*acc, low_elem);
        }
        for (&high_elem, acc) in high_remainder.iter().zip(accumulators.iter_mut()) {
            *acc = high_iter(*acc, high_elem);
        }
    }
    (accumulators, inputs)
}

/// Minimal optimization barrier needed to avoid accumulator autovectorization
///
/// If we accumulate data naively, LLVM will figure out that we're aggregating
/// arrays of inputs into identically sized arrays of accumulators and helpfully
/// autovectorize the accumulation. We do not want this optimization here
/// because it gets in the way of us separately studying the hardware overhead
/// of scalar and SIMD operations.
///
/// Therefore, we send accumulators to opaque inline ASM via `pessimize::hide()`
/// after ~each accumulation iteration so that LLVM is forced to run the
/// computations on the actual specified data type, or more precisely decides to
/// do so because the overhead of assembling data in wide vectors then
/// re-splitting it into narrow vectores is higher than the performance gain of
/// autovectorizing.
///
/// Unfortunately, such frequent optimization barriers come with side effects
/// like useless register-register moves and a matching increase in FP register
/// pressure. To reduce and ideally fully eliminate the impact, it is better to
/// send only a subset of the accumulators through the optimization barrier, and
/// to refrain from using it altogether in situations where autovectorization is
/// not known to be possible. This is what this function does.
#[inline]
fn hide_accumulators<T: FloatLike, const ILP: usize>(mut accumulators: [T; ILP]) -> [T; ILP] {
    // No need for pessimize::hide() optimization barriers if this accumulation
    // cannot be autovectorized because e.g. we are operating at the maximum
    // hardware-supported SIMD vector width.
    let Some(min_vector_ilp) = T::MIN_VECTORIZABLE_ILP else {
        return accumulators;
    };

    // Otherwise apply pessimize::hide() to a set of accumulators which is large
    // enough that there aren't enough remaining non-hidden accumulators left
    // for the compiler to perform autovectorization.
    //
    // If rustc/LLVM ever becomes crazy enough to autovectorize with masked SIMD
    // operations, then this -1 may need to become a /2 or worse, depending on
    // how efficient the compiler estimates that masked SIMD operations are on
    // the CPU target that the benchmark will run on. It's debatable whether
    // this is best handled here or in the definition of MIN_VECTORIZABLE_ILP.
    let max_elided_barriers = min_vector_ilp.get() - 1;
    let min_hidden_accs = ILP.saturating_sub(max_elided_barriers);
    for acc in accumulators.iter_mut().take(min_hidden_accs) {
        *acc = pessimize::hide::<T>(*acc);
    }
    accumulators
}

// --- Data abstraction layer ---

/// Floating-point type that can be natively processed by the CPU
trait FloatLike:
    Add<Output = Self>
    + AddAssign
    + Copy
    + Default
    + Div<Output = Self>
    + DivAssign
    + Mul<Output = Self>
    + MulAssign
    + Neg<Output = Self>
    + Pessimize
    + Sub<Output = Self>
    + SubAssign
{
    /// Random distribution of positive normal IEEE-754 floats
    ///
    /// Our normal inputs follow the distribution of 2.0.pow(u), where u follows
    /// a uniform distribution from -1 to 1. We use this distribution because it
    /// has several good properties:
    ///
    /// - The numbers are close to 1, which is the optimum range for
    ///   floating-point arithmetic:
    ///     * Starting from a value of order of magnitude 2^(MANTISSA_DIGITS/2),
    ///       we can randomly add and subtract numbers close to 1 for a long
    ///       time before we run into significant precision issues (due to the
    ///       accumulator becoming too large) or cancelation/underflow issues
    ///       (due to the accumulator becoming too small)
    ///     * Starting from a value of order of magnitude 1, we can multiply or
    ///       divide by values close to 1 for a long time before hitting
    ///       exponent overflow or underflow issues.
    /// - The particular distribution chosen between 0.5 to 2.0 additionally
    ///   ensures that repeatedly multiplying by numbers from this distribution
    ///   results in a random walk: an accumulator that is repeatedly multiplied
    ///   by such values should oscillate around its initial value in
    ///   multiplicative steps of at most * 2.0 or / 2.0 per iteration, with low
    ///   odds of getting too large or too small if the RNG is working
    ///   correctly. If the accumulator starts close to 1, we are well protected
    ///   from exponent overflow and underflow during this random walk.
    ///
    /// For SIMD types, we generate a vector of such floats.
    fn normal_sampler() -> impl Fn(&mut ThreadRng) -> Self;

    /// Random distribution of positive subnormal floats
    ///
    /// These values are effectively treated as zeros by all of our current
    /// benchmarks, therefore their exact distribution does not matter at this
    /// point in time. We should just strive to evenly cover the space of all
    /// possible subnormals, i.e. generate a subnormal value with uniformly
    /// distributed random mantissa bits.
    ///
    /// For SIMD types, we generate a vector of such floats.
    fn subnormal_sampler() -> impl Fn(&mut ThreadRng) -> Self;

    /// Fill up the `target` input data buffer with random input, taking
    /// `num_subnormals` values from `subnormal_sampler`, and the other values
    /// from `normal_sampler`
    ///
    /// Note that after running this, subnormal/normal inputs are not randomly
    /// distributed in the array. This ordering issue will be taken care of by
    /// `FloatSet::make_sequence()` in a later step of the pipeline.
    fn generate_positive_inputs(target: &mut [Self], num_subnormals: usize) {
        assert!(num_subnormals <= target.len());
        let mut rng = rand::thread_rng();
        let (subnormal_target, normal_target) = target.split_at_mut(num_subnormals);
        let subnormal = Self::subnormal_sampler();
        for target in subnormal_target {
            *target = subnormal(&mut rng);
        }
        let normal = Self::normal_sampler();
        for target in normal_target {
            *target = normal(&mut rng);
        }
    }

    /// Generate an array of inputs using `generate_positive_inputs`
    #[cfg(feature = "register_data_sources")]
    fn generate_positive_input_array<const N: usize>(num_subnormals: usize) -> [Self; N] {
        let mut result = [Self::default(); N];
        Self::generate_positive_inputs(&mut result[..], num_subnormals);
        result
    }

    /// Minimal number of accumulators of this type that the compiler needs to
    /// autovectorize the accumulation.
    ///
    /// `None` means that this accumulator type has the maximal supported width
    /// for this hardware. Therefore autovectorization is impossible and
    /// `pessimize::hide()` barriers on accumulators can be omitted.
    ///
    /// If we don't know for the active hardware, we return `Some(2)` as a safe
    /// default, which means that all but one accumulator will need to go
    /// through a `pessimize::hide()` optimization barrier.
    ///
    /// This is a workaround for rustc/LLVM spilling accumulators to memory when
    /// they are passed through `pessimize::hide()` in benchmarks that operates
    /// from memory inputs. See [`hide_accumulators()`] for more context.
    const MIN_VECTORIZABLE_ILP: Option<NonZeroUsize>;

    // We're also gonna need some float data & ops not exposed via std traits
    const MANTISSA_DIGITS: u32;
    fn splat(x: f32) -> Self;
    fn mul_add(self, factor: Self, addend: Self) -> Self;
    fn sqrt(self) -> Self;
}
//
impl FloatLike for f32 {
    fn normal_sampler() -> impl Fn(&mut ThreadRng) -> Self {
        let dist = Uniform::new(-1.0, 1.0);
        move |rng| 2.0f32.powf(rng.sample(dist))
    }

    fn subnormal_sampler() -> impl Fn(&mut ThreadRng) -> Self {
        let dist = Uniform::new(2.0f32.powi(-149), 2.0f32.powi(-126));
        move |rng| rng.sample(dist)
    }

    const MIN_VECTORIZABLE_ILP: Option<NonZeroUsize> = const {
        if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
            // Ignoring MMX, 3DNow!, and other legacy 64-bit x86 SIMD
            // instruction sets which are not supported by compilers anymore
            if cfg!(any(target_arch = "x86_64", target_feature = "sse")) {
                // rustc has been observed to generate code for half-vectors of
                // f32, likely because they can be moved using a single movsd
                NonZeroUsize::new(2)
            } else if cfg!(target_feature = "avx") {
                NonZeroUsize::new(8)
            } else if cfg!(target_feature = "avx512f") {
                NonZeroUsize::new(16)
            } else {
                None
            }
        } else {
            // TODO: Investigate other hardware
            NonZeroUsize::new(2)
        }
    };

    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;

    #[inline]
    fn splat(x: f32) -> Self {
        x
    }

    #[inline]
    fn mul_add(self, factor: Self, addend: Self) -> Self {
        self.mul_add(factor, addend)
    }

    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}
//
impl FloatLike for f64 {
    fn normal_sampler() -> impl Fn(&mut ThreadRng) -> Self {
        let dist = Uniform::new(-1.0, 1.0);
        move |rng| 2.0f64.powf(rng.sample(dist))
    }

    fn subnormal_sampler() -> impl Fn(&mut ThreadRng) -> Self {
        let dist = Uniform::new(2.0f64.powi(-1074), 2.0f64.powi(-1022));
        move |rng| rng.sample(dist)
    }

    const MIN_VECTORIZABLE_ILP: Option<NonZeroUsize> = const {
        if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
            if cfg!(any(target_arch = "x86_64", target_feature = "sse2")) {
                NonZeroUsize::new(2)
            } else if cfg!(target_feature = "avx") {
                NonZeroUsize::new(4)
            } else if cfg!(target_feature = "avx512f") {
                NonZeroUsize::new(8)
            } else {
                None
            }
        } else {
            // TODO: Investigate other hardware
            NonZeroUsize::new(2)
        }
    };

    const MANTISSA_DIGITS: u32 = f64::MANTISSA_DIGITS;

    #[inline]
    fn splat(x: f32) -> Self {
        x as f64
    }

    #[inline]
    fn mul_add(self, factor: Self, addend: Self) -> Self {
        self.mul_add(factor, addend)
    }

    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}
//
impl<const WIDTH: usize> FloatLike for Simd<f32, WIDTH>
where
    LaneCount<WIDTH>: SupportedLaneCount,
    Self: Pessimize + StdFloat,
{
    fn normal_sampler() -> impl Fn(&mut ThreadRng) -> Self {
        let sampler = f32::normal_sampler();
        move |rng| std::array::from_fn(|_| sampler(rng)).into()
    }

    fn subnormal_sampler() -> impl Fn(&mut ThreadRng) -> Self {
        let sampler = f32::subnormal_sampler();
        move |rng| std::array::from_fn(|_| sampler(rng)).into()
    }

    const MIN_VECTORIZABLE_ILP: Option<NonZeroUsize> = const {
        if WIDTH == 1 {
            f32::MIN_VECTORIZABLE_ILP
        } else if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
            match WIDTH {
                16 => {
                    assert!(cfg!(target_feature = "avx512f"));
                    None
                }
                8 => {
                    assert!(cfg!(target_feature = "avx"));
                    if cfg!(target_feature = "avx512f") {
                        NonZeroUsize::new(2)
                    } else {
                        None
                    }
                }
                4 => {
                    assert!(cfg!(any(target_arch = "x86_64", target_feature = "sse")));
                    if cfg!(target_feature = "avx") {
                        NonZeroUsize::new(2)
                    } else if cfg!(target_feature = "avx512f") {
                        NonZeroUsize::new(4)
                    } else {
                        None
                    }
                }
                _ => unreachable!(),
            }
        } else {
            // TODO: Investigate other hardware
            NonZeroUsize::new(2)
        }
    };

    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;

    #[inline]
    fn splat(x: f32) -> Self {
        Simd::splat(x)
    }

    #[inline]
    fn mul_add(self, factor: Self, addend: Self) -> Self {
        <Self as StdFloat>::mul_add(self, factor, addend)
    }

    #[inline]
    fn sqrt(self) -> Self {
        <Self as StdFloat>::sqrt(self)
    }
}
//
impl<const WIDTH: usize> FloatLike for Simd<f64, WIDTH>
where
    LaneCount<WIDTH>: SupportedLaneCount,
    Self: Pessimize + StdFloat,
{
    fn normal_sampler() -> impl Fn(&mut ThreadRng) -> Self {
        let sampler = f64::normal_sampler();
        move |rng| std::array::from_fn(|_| sampler(rng)).into()
    }

    fn subnormal_sampler() -> impl Fn(&mut ThreadRng) -> Self {
        let sampler = f64::subnormal_sampler();
        move |rng| std::array::from_fn(|_| sampler(rng)).into()
    }

    const MIN_VECTORIZABLE_ILP: Option<NonZeroUsize> = const {
        if WIDTH == 1 {
            f64::MIN_VECTORIZABLE_ILP
        } else if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
            match WIDTH {
                8 => {
                    assert!(cfg!(target_feature = "avx512f"));
                    None
                }
                4 => {
                    assert!(cfg!(target_feature = "avx"));
                    if cfg!(target_feature = "avx512f") {
                        NonZeroUsize::new(2)
                    } else {
                        None
                    }
                }
                2 => {
                    assert!(cfg!(any(target_arch = "x86_64", target_feature = "sse2")));
                    if cfg!(target_feature = "avx") {
                        NonZeroUsize::new(2)
                    } else if cfg!(target_feature = "avx512f") {
                        NonZeroUsize::new(4)
                    } else {
                        None
                    }
                }
                _ => unreachable!(),
            }
        } else {
            // TODO: Investigate other hardware
            NonZeroUsize::new(2)
        }
    };

    const MANTISSA_DIGITS: u32 = f64::MANTISSA_DIGITS;

    #[inline]
    fn splat(x: f32) -> Self {
        Simd::splat(x as f64)
    }

    #[inline]
    fn mul_add(self, factor: Self, addend: Self) -> Self {
        <Self as StdFloat>::mul_add(self, factor, addend)
    }

    #[inline]
    fn sqrt(self) -> Self {
        <Self as StdFloat>::sqrt(self)
    }
}

/// Unordered collection of benchmark inputs, to be ordered before use
///
/// For sufficiently small collections of inputs (register inputs being the
/// extreme case), the measured benchmark performance may depend on the position
/// of subnormal inputs in the input data sequence.
///
/// This is for example known to be the case for the `fma_full_average`
/// benchmark on current Intel CPUs: since the trigger for subnormal slowdown on
/// those CPUs is that at least one input to an instruction is subnormal, and
/// two subnormals inputs do not increase overhead, an input data configuration
/// for which half of the FMAs have all-normal inputs and half of the FMAs have
/// all-subnormal inputs should be twice as fast as another configuration where
/// all FMAs have one normal and one subnormal input.
///
/// Further, because the subnormal fallback of some hardware trashes the CPU
/// frontend, we could also expect a Sufficiently Weird CPU Microarchitecture to
/// have a subnormal-induced slowdown that varies depending on where in the
/// input program the affected instructions are located, e.g. how close they are
/// to the beginning or the end of a benchmark loop iteration in machine code.
///
/// This means that to make the measured benchmark performance as reproducible
/// as possible across `cargo bench` runs, we need to do two things:
///
/// - Initially ensure that the advertised share of subnormal inputs is exact,
///   not approximately achieved by relying on large-scale statistical behavior.
/// - Randomly shuffle inputs on each benchmark run so that the each instruction
///   in the program gets all possible subnormal/normal input configurations
///   evenly covered, given enough criterion benchmark runs.
///
/// The former is achieved by `T::generate_positive_inputs()`, while the latter
/// is achieved by an input reordering step between benchmark iteration batches.
/// The `FloatSet`/`FloatSequence` dichotomy enforces such a reordering step.
trait FloatSet<T: FloatLike>: AsMut<[T]> {
    /// Associated FloatSequence type
    type Sequence<'a>: FloatSequence<T>
    where
        Self: 'a;

    /// Generate a randomly ordered sequence of inputs, hidden from the
    /// optimizer and initially resident in CPU registers.
    fn make_sequence(&mut self) -> Self::Sequence<'_>;

    /// Re-expose useful FloatSequence properties for easier access
    const IS_REUSED: bool = Self::Sequence::<'_>::IS_REUSED;
    const NUM_REGISTER_INPUTS: Option<usize> = Self::Sequence::<'_>::NUM_REGISTER_INPUTS;
}
//
impl<T: FloatLike, const N: usize> FloatSet<T> for [T; N] {
    type Sequence<'a>
        = Self
    where
        Self: 'a;

    fn make_sequence(&mut self) -> Self {
        self.shuffle(&mut rand::thread_rng());
        <[T; N] as FloatSequence<T>>::hide(*self)
    }
}
//
impl<T: FloatLike> FloatSet<T> for &mut [T] {
    type Sequence<'a>
        = &'a [T]
    where
        Self: 'a;

    fn make_sequence(&mut self) -> &[T] {
        self.shuffle(&mut rand::thread_rng());
        pessimize::hide::<&[T]>(*self)
    }
}

/// Randomly ordered sequence of inputs that is ready for benchmark consumption
trait FloatSequence<T: FloatLike>: AsRef<[T]> + Copy {
    /// Pass each inner value through `pessimize::hide()`. This ensures that...
    ///
    /// - The returned values are unrelated to the original values in the eyes
    ///   of the optimizer. This is needed to avoids compiler over-optimization
    ///   of benchmarks that reuse a small set of inputs (e.g. hoisting of
    ///   `sqrt()` computations out of the iteration loop in
    ///   `sqrt_positive_addsub` benchmarks with register inputs).
    /// - Each inner value ends up confined in its own CPU register. This can
    ///   applied to the accumulator set between accumulation steps to avoid
    ///   unwanted compiler autovectorization of scalar accumulation code.
    fn hide(self) -> Self;

    /// CPU floating-point registers that are used by this input data
    ///
    /// None means that the input comes from memory (CPU cache or RAM).
    const NUM_REGISTER_INPUTS: Option<usize>;

    /// Truth that we must reuse the same input for each accumulator
    const IS_REUSED: bool;
}
//
impl<T: FloatLike, const N: usize> FloatSequence<T> for [T; N] {
    #[inline]
    fn hide(mut self) -> Self {
        for elem in self.iter_mut() {
            *elem = pessimize::hide::<T>(*elem);
        }
        self
    }

    const NUM_REGISTER_INPUTS: Option<usize> = Some(N);

    const IS_REUSED: bool = true;
}
//
impl<T: FloatLike> FloatSequence<T> for &[T] {
    #[inline]
    fn hide(self) -> Self {
        pessimize::hide::<&[T]>(self)
    }

    const NUM_REGISTER_INPUTS: Option<usize> = None;

    const IS_REUSED: bool = false;
}

// --- Other microarchitectural information ---

/// Lower bound on the number of architectural scalar/SIMD registers.
///
/// To exhaustively cover all hardware-allowed configurations, this should
/// ideally strive to be an exact count, not a lower bound, for all hardware of
/// highest interest to this benchmark's users.
///
/// However, in the case of hardware architectures like Arm that have a
/// complicated past and a fragmented present, accurately spelling this out for
/// all hardware that can be purchased today would be quite a bit of effort with
/// relatively little payoff. So we tolerate some lower bound approximation.
const MIN_FLOAT_REGISTERS: usize = const {
    let target = target_features::CURRENT_TARGET;
    match target.architecture() {
        Architecture::Arm | Architecture::AArch64 => {
            if target.supports_feature_str("sve") {
                32
            } else {
                16
            }
        }
        Architecture::X86 => {
            if target.supports_feature_str("avx512f") {
                32
            } else {
                16
            }
        }
        Architecture::RiscV => 32,
        // TODO: Check for other architectures
        _ => 16,
    }
};

/// Truth that the current hardware architecture is known to support memory
/// operands for scalar and SIMD operations.
///
/// This means that when we are doing benchmarks like `addsub` that directly
/// reduce memory inputs into accumulators, we don't need to load inputs into
/// CPU registers before reducing them into the accumulator. As a result, we can
/// use more CPU registers as accumulators on those benchmarks.
const HAS_MEMORY_OPERANDS: bool = const {
    let target = target_features::CURRENT_TARGET;
    match target.architecture() {
        Architecture::X86 => true,
        // TODO: Check for other architectures
        _ => false,
    }
};

// --- Criterion boilerplate ---

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
