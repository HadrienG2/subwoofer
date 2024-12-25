#![feature(portable_simd)]

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion, Throughput,
};
use hwlocality::Topology;
#[cfg(feature = "simd")]
use std::simd::Simd;
use std::time::Instant;
use subwoofer::{
    arch::{HAS_MEMORY_OPERANDS, MIN_FLOAT_REGISTERS},
    benchmarks::{self, MAX_SUBNORMAL_CONFIGURATIONS},
    types::{FloatLike, FloatSequence, FloatSet},
};

// --- Benchmark configuration and steering ---

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
            benchmarks::iter_halves::<_, _, ILP, false>(
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
                // Need to hide reused register inputs, so that the compiler
                // doesn't abusively factor out the redundant square root
                // computations and reuse their result for all accumulators (in
                // fact it would even be allowed to reuse them for the entire
                // outer iters loop in run_benchmark).
                benchmarks::iter_halves::<_, _, ILP, true>(
                    accumulators,
                    inputs,
                    low_iter,
                    high_iter,
                )
            } else {
                // Memory inputs do not need to be hidden because each
                // accumulator gets its own input substream (preventing square
                // root reuse during the inner loop over accumulators) and
                // current LLVM is not crazy enough to precompute square roots
                // for a whole arbitrarily large dynamically-sized batch of
                // input data.
                assert!(Inputs::NUM_REGISTER_INPUTS.is_none());
                benchmarks::iter_halves::<_, _, ILP, false>(
                    accumulators,
                    inputs,
                    low_iter,
                    high_iter,
                )
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
            benchmarks::iter_full(accumulators, inputs, |acc, elem| {
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
            benchmarks::iter_full(accumulators, inputs, move |acc, elem| {
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
            benchmarks::iter_full(accumulators, inputs, move |acc, elem| {
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
            benchmarks::iter_full(accumulators, inputs, move |acc, elem| {
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
                    accumulators = benchmarks::hide_accumulators(accumulators);
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
                    accumulators = benchmarks::hide_accumulators(accumulators);
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

// --- Criterion boilerplate ---

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
