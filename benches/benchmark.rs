#![feature(portable_simd)]
#![feature(stmt_expr_attributes)]

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
use hwlocality::Topology;
use pessimize::Pessimize;
use rand::{distributions::Uniform, prelude::*};
use std::{
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
                // no registers left for input.
                $benchmark::<$t, $inputregs> $args with { group: &mut group, selected_ilp: $type_config.ilp, instantiated_ilps: [1, 2, 4, 8, 16] }
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
                run_benchmark(
                    group,
                    inputs,
                    input_name,
                    additive_accumulator_init,
                    #[inline(always)]
                    |acc, inputs| addsub(acc, inputs),
                );
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
                    #[inline(always)]
                    |acc, inputs| sqrt_positive_addsub(acc, inputs),
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
                        #[inline(always)]
                        |acc, inputs| average(acc, inputs),
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
                        #[inline(always)]
                        |acc, inputs| mul_average(acc, average_target, inputs),
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
                        #[inline(always)]
                        |acc, inputs| fma_multiplier_average(acc, average_target, inputs),
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
                        #[inline(always)]
                        |acc, inputs| fma_addend_average(acc, average_target, inputs),
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
                    #[inline(always)]
                    |acc, inputs| fma_full_average(acc, average_target, inputs),
                );
            }
            break 'select_benchmark;
        }
    }
}

/// Run a criterion benchmark of the specified computation
#[inline(never)] // Make most of the configuration appear in perf profiles
fn run_benchmark<T: FloatLike, TSet: FloatSet<T>, const ILP: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    mut inputs: TSet,
    input_name: String,
    accumulator_init: [T; ILP],
    mut iteration: impl FnMut(&mut [T; ILP], TSet::Sequence<'_>) + Copy,
) {
    group.bench_function(input_name, move |b| {
        b.iter_custom(|iters| {
            // For each benchmark iteration batch, we make sure that...
            //
            // - The compiler cannot leverage the fact that the initial
            //   accumulators are always the same in order to eliminate
            //   "redundant" computations across iteration batches.
            // - Inputs are randomly reordered from one batch to another, which
            //   will avoid input order-related bias if criterion runs enough
            //   batches (as it does in its default configuration).
            // - Accumulators and register inputs are resident in CPU registers
            //   before the timer starts, eliminatign possible bias for small
            //   numbers of iterations vs large numbers of iterations.
            let mut accumulators = <[T; ILP] as FloatSequence<T>>::hide(accumulator_init);
            let inputs = inputs.make_sequence();

            // Timed region, be careful with optimization barriers here
            let start = Instant::now();
            for _ in 0..iters {
                // Note that there is no optimization barrier on inputs,
                // because the impact on codegen was observed to be too high
                // in the case of register inputs (lots of useless reg-reg
                // moves in generated ASM) and we don't really need them for
                // memory input (they contain too much data for the compiler
                // to be clever and optimize out iterations).
                //
                // This means that we must be careful about invalid
                // optimizations based on compiler-observed benchmark input
                // reuse inside of the microbenchmarks themselves.
                iteration(&mut accumulators, inputs);
            }
            let duration = start.elapsed();

            // Make the compiler think that the output of the computation is
            // used, so that the optimizer doesn't delete the computation.
            for acc in accumulators {
                pessimize::consume(acc);
            }
            duration
        });
    });
}

// --- Actual benchmarks ---

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
#[cfg(feature = "bench_addsub")]
#[inline(always)]
fn addsub<T: FloatLike, const ILP: usize>(
    accumulators: &mut [T; ILP],
    inputs: impl FloatSequence<T>,
) {
    iter_halves(
        accumulators,
        inputs,
        #[inline(always)]
        |acc, elem| *acc = pessimize::hide::<T>(*acc) + elem,
        #[inline(always)]
        |acc, elem| *acc = pessimize::hide::<T>(*acc) - elem,
    );
}

/// Benchmark square root of positive numbers, followed by add/sub cycle
///
/// Square roots of negative numbers may or may not be emulated in software.
/// They are thus not a good candidate for CPU microbenchmarking.
///
/// Square roots must be stored in one temporary register before add/sub,
/// therefore one CPU register must always be reserved for square root
/// temporaries, in addition to the registers used for inputs and accumulators.
#[cfg(feature = "bench_sqrt_positive_addsub")]
#[inline(always)]
fn sqrt_positive_addsub<T: FloatLike, TSeq: FloatSequence<T>, const ILP: usize>(
    accumulators: &mut [T; ILP],
    inputs: TSeq,
) {
    let hidden_sqrt = |elem: T| {
        if TSeq::IS_REUSED {
            // Need this optimization barrier in the case of reused register
            // inputs, so that the compiler doesn't abusively factor out the
            // square root computation and reuse the result for all accumulators
            // (or even for the entire computation)
            pessimize::hide::<T>(elem).sqrt()
        } else {
            // No need in the case of non-reused memory inputs: each accumulator
            // gets a different element from the input buffer to work with, and
            // current compilers are not crazy enough to precompute square roots
            // for a whole arbitrarily large batch of input data.
            assert!(TSeq::NUM_REGISTER_INPUTS.is_none());
            elem.sqrt()
        }
    };
    iter_halves(
        accumulators,
        inputs,
        #[inline(always)]
        |acc, elem| *acc = pessimize::hide::<T>(*acc) + hidden_sqrt(elem),
        #[inline(always)]
        |acc, elem| *acc = pessimize::hide::<T>(*acc) - hidden_sqrt(elem),
    );
}

/// For multiplicative benchmarks, we're going to need to an extra
/// averaging operation, otherwise once we've multiplied by a subnormal
/// we'll stay in subnormal range forever.
///
/// It cannot be just an addition, because otherwise if we have no subnormal
/// input we get unbounded growth, which is also a problem.
///
/// This benchmark measures the overhead of averaging with an in-register input
/// in isolation, so that it can be subtracted from the overhead of X +
/// averaging (with due respect paid to the existence of superscalar execution).
///
/// At least one CPU register must be reserved to the averaging weight. Then...
///
/// - If the input comes from registers, no further CPU register needs to
///   reserved because the initial accumulator register can be reused for (acc +
///   elem), and then the product of that by the averaging weight.
/// - If the input comes from memory, then we need one extra CPU register for
///   the input memory load on architectures without memory operands, before it
///   can be summed with the accumulator. This is not necessary on architectures
///   with memory operands.
#[cfg(feature = "bench_average")]
#[inline(always)]
fn average<T: FloatLike, const ILP: usize>(
    accumulators: &mut [T; ILP],
    inputs: impl FloatSequence<T>,
) {
    iter_full(
        accumulators,
        inputs,
        #[inline(always)]
        |acc, elem| *acc = (elem + pessimize::hide::<T>(*acc)) * T::splat(0.5),
    );
}

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
#[cfg(feature = "bench_mul_average")]
#[inline(always)]
fn mul_average<T: FloatLike, const ILP: usize>(
    accumulators: &mut [T; ILP],
    target: T,
    inputs: impl FloatSequence<T>,
) {
    iter_full(
        accumulators,
        inputs,
        #[inline(always)]
        move |acc, elem| *acc = ((pessimize::hide::<T>(*acc) * elem) + target) * T::splat(0.5),
    );
}

/// Benchmark FMA with a possibly subnormal multiplier, followed by averaging
#[cfg(feature = "bench_fma_multiplier_average")]
#[inline(always)]
fn fma_multiplier_average<T: FloatLike, const ILP: usize>(
    accumulators: &mut [T; ILP],
    target: T,
    inputs: impl FloatSequence<T>,
) {
    let halve_weight = T::splat(0.5);
    iter_full(
        accumulators,
        inputs,
        #[inline(always)]
        move |acc, elem| {
            *acc = (pessimize::hide::<T>(*acc).mul_add(elem, halve_weight) + target) * halve_weight;
        },
    );
}

/// Benchmark FMA with a possibly subnormal addend, folowed by averaging
#[cfg(feature = "bench_fma_addend_average")]
#[inline(always)]
fn fma_addend_average<T: FloatLike, const ILP: usize>(
    accumulators: &mut [T; ILP],
    target: T,
    inputs: impl FloatSequence<T>,
) {
    let halve_weight = T::splat(0.5);
    iter_full(
        accumulators,
        inputs,
        #[inline(always)]
        move |acc, elem| {
            *acc = (pessimize::hide::<T>(*acc).mul_add(halve_weight, elem) + target) * halve_weight;
        },
    );
}

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
#[cfg(feature = "bench_fma_full_average")]
#[inline(always)]
fn fma_full_average<T: FloatLike, TSeq: FloatSequence<T>, const ILP: usize>(
    accumulators: &mut [T; ILP],
    target: T,
    inputs: TSeq,
) {
    let mut local_accumulators = *accumulators;
    let inputs = inputs.as_ref();
    let (factor_inputs, addend_inputs) = inputs.split_at(inputs.len() / 2);
    let iter = #[inline(always)]
    |acc: &mut T, factor, addend| {
        *acc = (pessimize::hide::<T>(*acc).mul_add(factor, addend) + target) * T::splat(0.5);
    };
    if TSeq::IS_REUSED {
        assert_eq!(inputs.len() % 2, 0);
        for (&factor, &addend) in factor_inputs.iter().zip(addend_inputs) {
            for acc in local_accumulators.iter_mut() {
                iter(acc, factor, addend);
            }
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
                .zip(local_accumulators.iter_mut())
            {
                iter(acc, factor, addend);
            }
        }
        for ((&factor, &addend), acc) in factor_remainder
            .iter()
            .zip(addend_remainder)
            .zip(local_accumulators.iter_mut())
        {
            iter(acc, factor, addend);
        }
    }
    *accumulators = local_accumulators;
}

/// Benchmark skeleton that processes the full input identically
#[allow(unused)]
#[inline(always)]
fn iter_full<T: FloatLike, TSeq: FloatSequence<T>, const ILP: usize>(
    accumulators: &mut [T; ILP],
    inputs: TSeq,
    mut iter: impl FnMut(&mut T, T),
) {
    let mut local_accumulators = *accumulators;
    let inputs = inputs.as_ref();
    if TSeq::IS_REUSED {
        for &elem in inputs {
            for acc in local_accumulators.iter_mut() {
                iter(acc, elem);
            }
        }
    } else {
        let chunks = inputs.chunks_exact(ILP);
        let remainder = chunks.remainder();
        for chunk in chunks {
            for (&elem, acc) in chunk.iter().zip(local_accumulators.iter_mut()) {
                iter(acc, elem);
            }
        }
        for (&elem, acc) in remainder.iter().zip(local_accumulators.iter_mut()) {
            iter(acc, elem);
        }
    }
    *accumulators = local_accumulators;
}

/// Benchmark skeleton that treats halves of the input differently
#[allow(unused)]
#[inline(always)]
fn iter_halves<T: FloatLike, TSeq: FloatSequence<T>, const ILP: usize>(
    accumulators: &mut [T; ILP],
    inputs: TSeq,
    mut low_iter: impl FnMut(&mut T, T),
    mut high_iter: impl FnMut(&mut T, T),
) {
    let mut local_accumulators = *accumulators;
    let inputs = inputs.as_ref();
    let (low_inputs, high_inputs) = inputs.split_at(inputs.len() / 2);
    if TSeq::IS_REUSED {
        assert_eq!(inputs.len() % 2, 0);
        for (&low_elem, &high_elem) in low_inputs.iter().zip(high_inputs) {
            for acc in local_accumulators.iter_mut() {
                low_iter(acc, low_elem);
            }
            for acc in local_accumulators.iter_mut() {
                high_iter(acc, high_elem);
            }
        }
    } else {
        let low_chunks = low_inputs.chunks_exact(ILP);
        let high_chunks = high_inputs.chunks_exact(ILP);
        let low_remainder = low_chunks.remainder();
        let high_remainder = high_chunks.remainder();
        for (low_chunk, high_chunk) in low_chunks.zip(high_chunks) {
            for (&low_elem, acc) in low_chunk.iter().zip(local_accumulators.iter_mut()) {
                low_iter(acc, low_elem);
            }
            for (&high_elem, acc) in high_chunk.iter().zip(local_accumulators.iter_mut()) {
                high_iter(acc, high_elem);
            }
        }
        for (&low_elem, acc) in low_remainder.iter().zip(local_accumulators.iter_mut()) {
            low_iter(acc, low_elem);
        }
        for (&high_elem, acc) in high_remainder.iter().zip(local_accumulators.iter_mut()) {
            high_iter(acc, high_elem);
        }
    }
    *accumulators = local_accumulators;
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

    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;

    #[inline(always)]
    fn splat(x: f32) -> Self {
        x
    }

    #[inline(always)]
    fn mul_add(self, factor: Self, addend: Self) -> Self {
        self.mul_add(factor, addend)
    }

    #[inline(always)]
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

    const MANTISSA_DIGITS: u32 = f64::MANTISSA_DIGITS;

    #[inline(always)]
    fn splat(x: f32) -> Self {
        x as f64
    }

    #[inline(always)]
    fn mul_add(self, factor: Self, addend: Self) -> Self {
        self.mul_add(factor, addend)
    }

    #[inline(always)]
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

    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;

    #[inline(always)]
    fn splat(x: f32) -> Self {
        Simd::splat(x)
    }

    #[inline(always)]
    fn mul_add(self, factor: Self, addend: Self) -> Self {
        <Self as StdFloat>::mul_add(self, factor, addend)
    }

    #[inline(always)]
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

    const MANTISSA_DIGITS: u32 = f64::MANTISSA_DIGITS;

    #[inline(always)]
    fn splat(x: f32) -> Self {
        Simd::splat(x as f64)
    }

    #[inline(always)]
    fn mul_add(self, factor: Self, addend: Self) -> Self {
        <Self as StdFloat>::mul_add(self, factor, addend)
    }

    #[inline(always)]
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
    #[inline(always)]
    fn hide(self) -> Self {
        self.map(pessimize::hide::<T>)
    }

    const NUM_REGISTER_INPUTS: Option<usize> = Some(N);

    const IS_REUSED: bool = true;
}
//
impl<T: FloatLike> FloatSequence<T> for &[T] {
    #[inline(always)]
    fn hide(self) -> Self {
        pessimize::hide::<&[T]>(self)
    }

    const NUM_REGISTER_INPUTS: Option<usize> = None;

    const IS_REUSED: bool = false;
}

// --- Microarchitectural information ---

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
