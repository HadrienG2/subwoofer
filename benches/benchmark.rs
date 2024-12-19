#![feature(portable_simd)]

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, Bencher, BenchmarkGroup, Criterion, Throughput};
use hwlocality::Topology;
use pessimize::Pessimize;
use rand::{distributions::Uniform, prelude::*};
use std::{
    borrow::Borrow,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    simd::{LaneCount, Simd, StdFloat, SupportedLaneCount},
    time::Instant,
};
use target_features::Architecture;

// --- Benchmark configuration and combinatorial explosion ---

/// Maximum granularity of subnormal occurence probabilities
///
/// Higher is more precise, but the benchmark execution time in the default
/// configuration is multipled accordingly.
const MAX_SUBNORMAL_CONFIGURATIONS: usize = const {
    if cfg!(feature = "more_subnormal_frequencies") {
        16 // 6.25% granularity
    } else {
        4 // 25% granularity
    }
};

/// Benchmark entry point
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

    // Then benchmark for each supported floating-point type
    let config = CommonConfiguration {
        benchmark_names: &benchmark_names,
        memory_input_sizes: &memory_input_sizes,
    };
    benchmark_type::<f32>(c, config, "f32");
    benchmark_type::<f64>(c, config, "f64");
    #[cfg(feature = "simd")]
    {
        // FIXME: Mess around with cfg_if to better approximate actually
        //        supported SIMD instruction sets on non-x86 architectures
        // Leading zeros works around poor criterion bench name sorting
        benchmark_type::<Simd<f32, 4>>(c, config, "f32x04");
        benchmark_type::<Simd<f64, 2>>(c, config, "f64x02");
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        {
            benchmark_type::<Simd<f32, 8>>(c, config, "f32x08");
            benchmark_type::<Simd<f64, 4>>(c, config, "f64x04");
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            benchmark_type::<Simd<f32, 16>>(c, config, "f32x16");
            benchmark_type::<Simd<f64, 8>>(c, config, "f64x08");
        }
    }
}

/// Common benchmark configuration
#[derive(Copy, Clone)]
struct CommonConfiguration<'a> {
    /// Operations that are being benchmarked
    benchmark_names: &'a [&'static str],

    /// Memory input sizes (in bytes) that are being benchmarked
    memory_input_sizes: &'a [usize],
}

/// Benchmark a set of ILP configurations for a given scalar/SIMD type, using
/// input data from a CPU cache or RAM
macro_rules! for_each_ilp {
    // Decide which ILP configurations we are going to try...
    ( $benchmark:ident ::< $t:ty > $args:tt with { config: $config:expr } ) => {
        for_each_ilp!(
            $benchmark::<$t> $args with { config: $config, ilp: [1, 2, 4, 8, 16, 32] }
        );
    };

    // ...then generate all these ILP configurations, pick the one currently
    // selected by the outer ILP loop in benchmark_type(), and make sure that
    // this configuration can fit in CPU registers for at least one benchmark...
    ( $benchmark:ident ::< $t:ty > $args:tt with { config: $config:expr, ilp: [ $($ilp:literal),* ] } ) => {
        // On CPU architectures without memory operands, we need one extra CPU
        // register to load memory inputs even in the addsub benchmark.
        let input_operand_registers = if HAS_MEMORY_OPERANDS { 0 } else { 1 };
        match $config.ilp {
            $(
                $ilp => if $ilp <= MIN_FLOAT_REGISTERS - input_operand_registers {
                    for_each_ilp!(
                        $benchmark::<$t, $ilp> $args with { benchmark_name: $config.benchmark_name }
                    );
                }
            )*
            _ => unimplemented!("Asked to run with unsupported ILP {}", $config.ilp),
        }
    };

    // ...finally, add the benchmark name to the list of benchmark arguments and
    // call the inner benchmark worker.
    ( $benchmark:ident ::< $t:ty, $ilp:literal >( $($arg:expr),* ) with { benchmark_name: $benchmark_name:expr } ) => {
        $benchmark::<$t, $ilp>( $($arg,)* $benchmark_name );
    };
}

/// Benchmark a set of ILP configurations for a given scalar/SIMD type, using
/// input from CPU registers
#[cfg(feature = "register_data_sources")]
macro_rules! for_each_registers_and_ilp {
    // Decide which INPUT_REGISTERS configurations we are going to try...
    ( $benchmark:ident ::< $t:ty > $args:tt with { config: $config:expr } ) => {
        #[cfg(feature = "tiny_inputs")]
        for_each_registers_and_ilp!(
            $benchmark::<$t> $args with { config: $config, inputregs: [1, 2, 4, 8, 16] }
        );
        #[cfg(not(feature = "tiny_inputs"))]
        for_each_registers_and_ilp!(
            // We need 1 register per accumulator and the highest ILP we target
            // is 16 (see below why). This ILP only works for platforms with 32
            // registers, as it leaves no register left for input on platforms
            // with 16 registers.
            //
            // On platforms with 16 registers, the highest we can go is ILP8,
            // which consumes 8 registers for accumulators and leaves 8 free for
            // inputs and temporaries like operands or square roots.
            //
            // 3-4 temporaries is enough for all the computations we're doing
            // here, therefore, configurations with <4 inputs underuse registers
            // on all known CPUs. We thus don't compile them in by default to
            // save compilation time.
            //
            // Similarly, >=32 input registers make no sense because the largest
            // platforms have 32 registers and we need some for accumulators
            $benchmark::<$t> $args with { config: $config, inputregs: [4, 8, 16] }
        );
    };

    // ...then loop over these INPUT_REGISTERS configurations, giving each a
    // human-readable name that will be added to the list of benchmark
    // arguments, then decide which ILP configurations we are going to try...
    ( $benchmark:ident ::< $t:ty > $args:tt with { config: $config:expr, inputregs: [ $($inputregs:literal),* ] } ) => {
        $({
            // Name this input configuration
            let data_source_name = if $inputregs == 1 {
                // Leading zeros works around poor criterion bench name sorting
                "01register".to_string()
            } else {
                format!("{:02}registers", $inputregs)
            };

            // Dispatch to ILP configurations
            for_each_registers_and_ilp!(
                // We need 1 register per accumulator and current hardware has
                // at most 32 scalar/SIMD registers, so it does not make sense
                // to compile for ILP >= 32 at this point in time as we'd have
                // no registers left for input.
                $benchmark::<$t, $inputregs> $args with { config: $config, data_source_name: &data_source_name, ilp: [1, 2, 4, 8, 16] }
            );
        })*
    };

    // ...then generate all these ILP configurations, pick the one currently
    // selected by the outer ILP loop in benchmark_type(), and make sure that
    // this configuration can fit in CPU registers for at least one benchmark...
    ( $benchmark:ident ::< $t:ty, $inputregs:literal > $args:tt with { config: $config:expr, data_source_name: $data_source_name:expr, ilp: [ $($ilp:literal),* ] } ) => {
        match $config.ilp {
            $(
                // Minimal register footprint = $inputregs shared inputs + $ilp accumulators
                $ilp => if ($ilp + $inputregs) <= MIN_FLOAT_REGISTERS {
                    for_each_registers_and_ilp!(
                        $benchmark::<$t, $inputregs, $ilp> $args with { data_source_name: $data_source_name, benchmark_name: $config.benchmark_name }
                    );
                }
            )*
            _ => unimplemented!("Asked to run with unsupported ILP {}", $config.ilp),
        }
    };

    // ...finally, add the data source name and benchmark name to the list of
    // benchmark arguments and call the inner benchmark worker.
    ( $benchmark:ident ::< $t:ty, $inputregs:literal, $ilp:literal >( $($arg:expr),* ) with { data_source_name: $data_source_name:expr, benchmark_name: $benchmark_name:expr } ) => {
        $benchmark::<$t, $inputregs, $ilp>( $($arg,)* $data_source_name, $benchmark_name );
    };
}

/// Benchmark all ILP configurations for a given scalar/SIMD type
#[inline(never)] // Trying to make perf profiles look nicer
fn benchmark_type<T: Input>(c: &mut Criterion, common_config: CommonConfiguration, tname: &str) {
    // For each benchmarked operation...
    for benchmark_name in common_config.benchmark_names {
        // ...and for each supported degree of ILP...
        for ilp in (0..32usize.ilog2()).map(|ilp_pow2| 2usize.pow(ilp_pow2)) {
            // Set up a criterion group for the (type, benchmark, ilp) triplet
            let ilp_name = if ilp == 1 {
                "chained".to_string()
            } else {
                // Leading zeros works around poor criterion bench name sorting
                format!("ilp{ilp:02}")
            };
            let mut group = c.benchmark_group(format!("{tname}/{benchmark_name}/{ilp_name}"));
            let type_config = TypeConfiguration {
                benchmark_name,
                ilp,
            };

            // Benchmark with input data that fits in CPU registers
            #[cfg(feature = "register_data_sources")]
            for_each_registers_and_ilp!(benchmark_ilp_registers::<T>(&mut group) with { config: type_config });

            // Benchmark with input data that fits in L1, L2, ... all the way to RAM
            for (idx, &input_size) in common_config.memory_input_sizes.iter().enumerate() {
                // Allocate required storage
                let num_elems = input_size / std::mem::size_of::<T>();
                let mut input_storage = vec![T::default(); num_elems];

                // Name this input configuration
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

                // Run the benchmarks at each supported ILP level
                for_each_ilp!(benchmark_ilp_memory::<T>(&mut group, &mut input_storage, &data_source_name) with { config: type_config });
            }
        }
    }
}

/// Type-specific benchmark configuration
#[derive(Copy, Clone)]
struct TypeConfiguration {
    /// Selected benchmark
    benchmark_name: &'static str,

    /// Selected degree of ILP
    ilp: usize,
}

/// Benchmark all scalar or SIMD configurations for a floating-point type, using
/// inputs from CPU registers
#[cfg(feature = "register_data_sources")]
#[inline(never)] // Trying to make perf profiles look nicer
fn benchmark_ilp_registers<T: Input, const INPUT_REGISTERS: usize, const ILP: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    data_source_name: &str,
    benchmark_name: &'static str,
) {
    // Iterate over subnormal configurations
    let num_subnormal_configurations = INPUT_REGISTERS.min(MAX_SUBNORMAL_CONFIGURATIONS);
    for subnormal_share in 0..=num_subnormal_configurations {
        // Generate input data
        let num_subnormals = subnormal_share * INPUT_REGISTERS / num_subnormal_configurations;
        let inputs = T::generate_positive_inputs_exact::<INPUT_REGISTERS>(num_subnormals);

        // Name this input
        let input_name = format!(
            "{data_source_name}/{num_subnormals:0num_digits$}in{INPUT_REGISTERS}_subnormal",
            // Leading zeros works around poor criterion bench name sorting
            num_digits = INPUT_REGISTERS.ilog10() as usize
        );

        // Run all the benchmarks on this input
        benchmark_ilp::<_, _, ILP>(group, benchmark_name, inputs, &input_name);
    }
}

/// Benchmark all scalar or SIMD configurations for a floating-point type, using
/// inputs from memory
#[inline(never)] // Trying to make perf profiles look nicer
fn benchmark_ilp_memory<T: Input, const ILP: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    input_storage: &mut [T],
    data_source_name: &str,
    benchmark_name: &'static str,
) {
    // Iterate over subnormal configurations
    for subnormal_probability in 0..=MAX_SUBNORMAL_CONFIGURATIONS {
        // Generate input data
        let subnormal_probability =
            subnormal_probability as f32 / MAX_SUBNORMAL_CONFIGURATIONS as f32;
        T::generate_positive_inputs(input_storage, subnormal_probability);

        // Name this input
        let input_name = format!(
            // Leading zeros works around poor criterion bench name sorting
            "{data_source_name}/{:03.0}%_subnormal",
            subnormal_probability * 100.0
        );

        // Run all the benchmarks on this input
        benchmark_ilp::<_, _, ILP>(group, benchmark_name, &*input_storage, &input_name);
    }
}

/// Benchmark all scalar or SIMD configurations for a floating-point type, using
/// inputs from CPU registers or memory
#[inline(never)] // Trying to make perf profiles look nicer
fn benchmark_ilp<T: Input, Inputs: InputSet<T>, const ILP: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    benchmark_name: &'static str,
    inputs: Inputs,
    input_name: &str,
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
    let num_inputs = inputs.as_ref().len();
    assert!(num_inputs >= 1);
    let num_operation_inputs = if Inputs::IS_REUSED {
        num_inputs * ILP // Each input is sent to each ILP stream
    } else {
        num_inputs // Each ILP stream gets its own substream of the input
    } as u64;
    group.throughput(Throughput::Elements(num_operation_inputs));

    // Check if we should benchmark a certain ILP configuration, knowing how
    // much registers besides accumulators and register inputs are consumed when
    // operating from register and memory inputs.
    let should_check_ilp = |aux_registers_regop: usize, aux_registers_memop: usize| {
        // First eliminate configurations that would spill to the stack
        let non_accumulator_registers = if let Some(input_registers) = Inputs::REGISTER_FOOTPRINT {
            input_registers + aux_registers_regop
        } else {
            aux_registers_memop
        };
        if ILP + non_accumulator_registers > MIN_FLOAT_REGISTERS {
            return false;
        }

        // Unless the more_ilp_configurations feature is turned on, also
        // eliminate configurations other than the minimum, maximum, and
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
                group.bench_with_input(
                    input_name,
                    &inputs,
                    make_criterion_benchmark(additive_accumulator_init, addsub),
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
                group.bench_with_input(
                    input_name,
                    &inputs,
                    make_criterion_benchmark(additive_accumulator_init, sqrt_positive_addsub),
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
                    group.bench_with_input(
                        input_name,
                        &inputs,
                        make_criterion_benchmark(averaging_accumulator_init, average),
                    );
                    break 'select_benchmark;
                }

                // Benchmark multiplication -> averaging
                #[cfg(feature = "bench_mul_average")]
                if benchmark_name == "mul_average" {
                    group.bench_with_input(
                        input_name,
                        &inputs,
                        make_criterion_benchmark(
                            averaging_accumulator_init,
                            #[inline(always)]
                            move |acc, inputs| mul_average(acc, average_target, inputs),
                        ),
                    );
                    break 'select_benchmark;
                }

                // Benchmark fma with possibly subnormal multiplier -> averaging
                #[cfg(feature = "bench_fma_multiplier_average")]
                if benchmark_name == "fma_multiplier_average" {
                    group.bench_with_input(
                        input_name,
                        &inputs,
                        make_criterion_benchmark(
                            averaging_accumulator_init,
                            #[inline(always)]
                            move |acc, inputs| fma_multiplier_average(acc, average_target, inputs),
                        ),
                    );
                    break 'select_benchmark;
                }

                // Benchmark fma with possibly subnormal addend -> averaging
                #[cfg(feature = "bench_fma_addend_average")]
                if benchmark_name == "fma_addend_average" {
                    group.bench_with_input(
                        input_name,
                        &inputs,
                        make_criterion_benchmark(
                            averaging_accumulator_init,
                            #[inline(always)]
                            move |acc, inputs| fma_addend_average(acc, average_target, inputs),
                        ),
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
            if num_inputs >= 2
                && should_check_ilp(
                    aux_registers_except_op1,
                    aux_registers_except_op1 + aux_registers_direct_memop,
                )
            {
                // Benchmark fma then averaging with possibly subnormal operands
                group.throughput(Throughput::Elements(num_operation_inputs / 2));
                group.bench_with_input(
                    input_name,
                    &inputs,
                    make_criterion_benchmark(
                        averaging_accumulator_init,
                        #[inline(always)]
                        move |acc, inputs| fma_full_average(acc, average_target, inputs),
                    ),
                );
            }
            break 'select_benchmark;
        }
    }
}

/// Wrap a benchmark for consumption by criterion's bench_with_input
fn make_criterion_benchmark<T: Input, Inputs: InputSet<T>, const ILP: usize>(
    accumulator_init: [T; ILP],
    mut iteration: impl FnMut(&mut [T; ILP], Inputs) + Copy,
) -> impl for<'a> FnMut(&mut Bencher<'a, WallTime>, &Inputs) {
    move |b, &inputs| {
        b.iter_custom(
            #[inline(always)]
            move |iters| {
                // This makes sure that the compiler treats each benchmark as
                // its own computation (no abusive caching), and also has the
                // beneficial side-effect of ensuring inputs and accumulators
                // are initially in CPU registers, which hints the compiler into
                // keeping them there.
                let mut accumulators = accumulator_init.hide();
                let inputs = inputs.hide();

                // Timed benchmark loop
                let start = Instant::now();
                for _ in 0..iters {
                    // Note that there is no optimization barrier on inputs,
                    // because the impact on codegen was observed to be too high
                    // when operating on register inputs (lots of useless
                    // reg-reg moves).
                    //
                    // This means that we must be careful about invalid
                    // optimizations based on benchmark input reuse in the
                    // microbenchmarks themselves.
                    iteration(&mut accumulators, inputs);
                }
                let duration = start.elapsed();

                // Make the compiler think that the output of the computation is
                // used, so that it does not delete the entire computation.
                for acc in accumulators {
                    pessimize::consume(acc);
                }
                duration
            },
        );
    }
}

// --- Actual benchmarks ---

/// Benchmark addition and subtraction
///
/// This is just a random additive walk of ~unity or subnormal step, so given a
/// high enough starting point, an initially normal accumulator should stay in
/// the normal range forever.
#[cfg(feature = "bench_addsub")]
#[inline(always)]
fn addsub<T: Input, const ILP: usize>(accumulators: &mut [T; ILP], inputs: impl InputSet<T>) {
    iter_halves(
        accumulators,
        inputs,
        |acc, elem| *acc += elem,
        |acc, elem| *acc -= elem,
    );
}

/// Benchmark square root of positive numbers, followed by add/sub cycle
///
/// Square roots of negative numbers may or may not be emulated in software.
/// They are thus not a good candidate for CPU microbenchmarking.
///
/// Square roots must be stored in one temporary register before add/sub,
/// therefore this computation consumes at least one extra register.
#[cfg(feature = "bench_sqrt_positive_addsub")]
#[inline(always)]
fn sqrt_positive_addsub<T: Input, Inputs: InputSet<T>, const ILP: usize>(
    accumulators: &mut [T; ILP],
    inputs: Inputs,
) {
    let hidden_sqrt = |elem: T| {
        if Inputs::IS_REUSED {
            // Need this optimization barrier in the case of reused register
            // inputs, so that the compiler doesn't abusively factor out the
            // square root computation and reuse the result for all accumulators
            // (or even for the entire computation)
            pessimize::hide(elem).sqrt()
        } else {
            // No need in the case of non-reused memory inputs: each accumulator
            // gets a different element from the input buffer to work with, and
            // current compilers are not crazy enough to precompute square roots
            // for a whole arbitrarily large batch of input data.
            assert!(Inputs::REGISTER_FOOTPRINT.is_none());
            elem.sqrt()
        }
    };
    iter_halves(
        accumulators,
        inputs,
        |acc, elem| *acc += hidden_sqrt(elem),
        |acc, elem| *acc -= hidden_sqrt(elem),
    );
}

/// For multiplicative benchmarks, we're going to need to an extra
/// averaging operation, otherwise once we've multiplied by a subnormal
/// we'll stay in subnormal range forever.
///
/// It cannot be just an addition, because otherwise if we have no subnormal
/// input we get unbounded growth, which is also a problem.
///
/// This benchmark measures the overhead of averaging in isolation, so that it
/// can be compared to the overhead of X + averaging (with due respect paid to
/// the existence of superscalar execution, of course).
///
/// Averaging uses 2 CPU registers, restricting available ILP
#[cfg(feature = "bench_average")]
#[inline(always)]
fn average<T: Input, const ILP: usize>(accumulators: &mut [T; ILP], inputs: impl InputSet<T>) {
    iter_full(accumulators, inputs, |acc, elem| {
        *acc = (*acc + elem) * T::splat(0.5)
    });
}

/// Benchmark multiplication followed by averaging
#[cfg(feature = "bench_mul_average")]
#[inline(always)]
fn mul_average<T: Input, const ILP: usize>(
    accumulators: &mut [T; ILP],
    target: T,
    inputs: impl InputSet<T>,
) {
    iter_full(accumulators, inputs, move |acc, elem| {
        *acc = ((*acc * elem) + target) * T::splat(0.5)
    });
}

/// Benchmark FMA with a possibly subnormal multiplier, followed by averaging
#[cfg(feature = "bench_fma_multiplier_average")]
#[inline(always)]
fn fma_multiplier_average<T: Input, const ILP: usize>(
    accumulators: &mut [T; ILP],
    target: T,
    inputs: impl InputSet<T>,
) {
    let halve_weight = T::splat(0.5);
    iter_full(accumulators, inputs, move |acc, elem| {
        *acc = (acc.mul_add(elem, halve_weight) + target) * halve_weight;
    });
}

/// Benchmark FMA with a possibly subnormal addend, folowed by averaging
#[cfg(feature = "bench_fma_addend_average")]
#[inline(always)]
fn fma_addend_average<T: Input, const ILP: usize>(
    accumulators: &mut [T; ILP],
    target: T,
    inputs: impl InputSet<T>,
) {
    let halve_weight = T::splat(0.5);
    iter_full(accumulators, inputs, move |acc, elem| {
        *acc = (acc.mul_add(halve_weight, elem) + target) * halve_weight;
    });
}

/// Benchmark FMA with possibly subnormal inputs, followed by averaging.
///
/// This benchmark has twice the register pressure (need registers for both the
/// accumulator and one of the operands because no hardware architecture that I
/// know of has an FMA variant with two memory operands), therefore it is
/// available in less ILP configurations.
#[cfg(feature = "bench_fma_full_average")]
#[inline(always)]
fn fma_full_average<T: Input, Inputs: InputSet<T>, const ILP: usize>(
    accumulators: &mut [T; ILP],
    target: T,
    inputs: Inputs,
) {
    let mut local_accumulators = *accumulators;
    let inputs = inputs.as_ref();
    let (factor_inputs, addend_inputs) = inputs.split_at(inputs.len() / 2);
    let iter = |acc: &mut T, factor, addend| {
        *acc = (acc.mul_add(factor, addend) + target) * T::splat(0.5);
    };
    if Inputs::IS_REUSED {
        for (&factor, &addend) in factor_inputs.iter().zip(addend_inputs) {
            for acc in local_accumulators.iter_mut() {
                iter(acc, factor, addend);
            }
            // Need this barrier to prevent autovectorization
            local_accumulators = local_accumulators.hide();
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
            // Need this barrier to prevent autovectorization
            local_accumulators = local_accumulators.hide();
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
fn iter_full<T: Input, Inputs: InputSet<T>, const ILP: usize>(
    accumulators: &mut [T; ILP],
    inputs: Inputs,
    mut iter: impl FnMut(&mut T, T),
) {
    let mut local_accumulators = *accumulators;
    let inputs = inputs.as_ref();
    if Inputs::IS_REUSED {
        for &elem in inputs {
            for acc in local_accumulators.iter_mut() {
                iter(acc, elem);
            }
        }
        // Need this barrier to prevent autovectorization
        local_accumulators = local_accumulators.hide();
    } else {
        let chunks = inputs.chunks_exact(ILP);
        let remainder = chunks.remainder();
        for chunk in chunks {
            for (&elem, acc) in chunk.iter().zip(local_accumulators.iter_mut()) {
                iter(acc, elem);
            }
            // Need this barrier to prevent autovectorization
            local_accumulators = local_accumulators.hide();
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
fn iter_halves<T: Input, Inputs: InputSet<T>, const ILP: usize>(
    accumulators: &mut [T; ILP],
    inputs: Inputs,
    mut low_iter: impl FnMut(&mut T, T),
    mut high_iter: impl FnMut(&mut T, T),
) {
    let mut local_accumulators = *accumulators;
    let inputs = inputs.as_ref();
    let (low_inputs, high_inputs) = inputs.split_at(inputs.len() / 2);
    if Inputs::IS_REUSED {
        for (&low_elem, &high_elem) in low_inputs.iter().zip(high_inputs) {
            for acc in local_accumulators.iter_mut() {
                low_iter(acc, low_elem);
            }
            for acc in local_accumulators.iter_mut() {
                high_iter(acc, high_elem);
            }
            // Need this barrier to prevent autovectorization
            local_accumulators = local_accumulators.hide();
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
            // Need this barrier to prevent autovectorization
            local_accumulators = local_accumulators.hide();
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

/// Array/slice abstraction layer
trait InputSet<T: Input>: AsRef<[T]> + Borrow<[T]> + Copy {
    /// Make a copy that is unrelated in the eyes of the optimizer
    fn hide(self) -> Self;

    /// Floating-point registers that are reserved for this input
    ///
    /// None means that the input comes from cache or RAM.
    const REGISTER_FOOTPRINT: Option<usize>;

    /// Truth that we must use the same input for each accumulator
    const IS_REUSED: bool;
}
//
impl<T: Input, const N: usize> InputSet<T> for [T; N] {
    #[inline(always)]
    fn hide(self) -> Self {
        self.map(pessimize::hide)
    }

    const REGISTER_FOOTPRINT: Option<usize> = Some(N);

    const IS_REUSED: bool = true;
}
//
impl<T: Input> InputSet<T> for &[T] {
    #[inline(always)]
    fn hide(self) -> Self {
        pessimize::hide(self)
    }

    const REGISTER_FOOTPRINT: Option<usize> = None;

    const IS_REUSED: bool = false;
}

/// f32/f64 and Scalar/SIMD abstraction layer
trait Input:
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
    ///   by such values should oscillates around its initial value in
    ///   multiplicative steps of at most * 2.0 or / 2.0 per iteration, with low
    ///   odds of getting very large or very small if the RNG is working
    ///   correctly. If the accumulator starts close to 1, we are best protected
    ///   from exponent overflow and underflow during this walk.
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

    /// Fill up `target` buffer with positive values that have probability
    /// `subnormal_probability` of being taken from `subnormal_sampler`, and are
    /// otherwise taken from `normal_sampler`.
    fn generate_positive_inputs(target: &mut [Self], subnormal_probability: f32) {
        let mut rng = rand::thread_rng();
        let normal = Self::normal_sampler();
        let subnormal = Self::subnormal_sampler();
        for target in target {
            *target = if rng.gen::<f32>() < subnormal_probability {
                subnormal(&mut rng)
            } else {
                normal(&mut rng)
            };
        }
    }

    /// Generate an array of values of which exactly `num_subnormals` values are
    /// taken from `subnormal_sampler`. The remaining values are taken from
    /// `normal_sampler` instead.
    ///
    /// The subnormal values are randomly distributed in the output array.
    #[cfg(feature = "register_data_sources")]
    fn generate_positive_inputs_exact<const N: usize>(num_subnormals: usize) -> [Self; N] {
        let normal = Self::normal_sampler();
        let subnormal = Self::subnormal_sampler();
        let mut rng = rand::thread_rng();
        let mut result = std::array::from_fn(|i| {
            if i < num_subnormals {
                subnormal(&mut rng)
            } else {
                normal(&mut rng)
            }
        });
        result.shuffle(&mut rng);
        result
    }

    // We're also gonna need some float data & ops not exposed via std traits
    const MANTISSA_DIGITS: u32;
    fn splat(x: f32) -> Self;
    fn mul_add(self, factor: Self, addend: Self) -> Self;
    fn sqrt(self) -> Self;
}
//
impl Input for f32 {
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
impl Input for f64 {
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
impl<const WIDTH: usize> Input for Simd<f32, WIDTH>
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
impl<const WIDTH: usize> Input for Simd<f64, WIDTH>
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

// --- Microarchitectural information ---

/// How many scalar/SIMD registers we can use before spilling
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
        _ => 16,
    }
};

/// Whether the current hardware architecture is known to support memory
/// operands for scalar and SIMD operations.
///
/// This reduces register pressure, as we don't need to load inputs into
/// registers before reducing them into the accumulator.
const HAS_MEMORY_OPERANDS: bool = const {
    let target = target_features::CURRENT_TARGET;
    match target.architecture() {
        Architecture::X86 => true,
        // TODO: Check for Arm, RISCV, etc.
        _ => false,
    }
};

// --- Criterion boilerplate ---

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
