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

// --- Benchmark configuration and combinatorics ---

/// Maximum granularity of subnormal occurence probabilities
///
/// Higher is more precise, but the benchmark execution time in the default
/// configuration is multipled accordingly.
const MAX_SUBNORMAL_CONFIGURATIONS: usize = const {
    if cfg!(feature = "more_subnormal_frequencies") {
        20 // 5% granularity
    } else {
        4 // 25% granularity
    }
};

pub fn criterion_benchmark(c: &mut Criterion) {
    // Probe an amount of data that fits in L1, L2, ... and add a size that only
    // fits in RAM for completeness
    let cache_stats = Topology::new().unwrap().cpu_cache_stats().unwrap();
    let smallest_data_cache_sizes = cache_stats.smallest_data_cache_sizes();
    let max_size_to_fit = |cache_size: u64| cache_size / 2;
    let min_size_to_overflow = |cache_size: u64| cache_size * 8;
    let interesting_data_sizes = if cfg!(feature = "more_memory_data_sources") {
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

    // Run benchmark for all supported data types
    benchmark_type::<f32>(c, "f32", &interesting_data_sizes);
    benchmark_type::<f64>(c, "f64", &interesting_data_sizes);
    #[cfg(feature = "simd")]
    {
        // FIXME: Mess around with cfg_if to express actual supported SIMD
        //        instruction sets on non-x86 architectures
        benchmark_type::<Simd<f32, 4>>(c, "f32x4", &interesting_data_sizes);
        benchmark_type::<Simd<f64, 2>>(c, "f64x2", &interesting_data_sizes);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        {
            benchmark_type::<Simd<f32, 8>>(c, "f32x8", &interesting_data_sizes);
            benchmark_type::<Simd<f64, 4>>(c, "f64x4", &interesting_data_sizes);
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            benchmark_type::<Simd<f32, 16>>(c, "f32x16", &interesting_data_sizes);
            benchmark_type::<Simd<f64, 8>>(c, "f64x8", &interesting_data_sizes);
        }
    }
}

/// Benchmark a set of ILP configurations for a given scalar/SIMD type, using
/// input from memory
macro_rules! for_each_ilp {
    ( $benchmark:ident ::< $t:ty > $args:tt ) => {
        for_each_ilp!( $benchmark::<$t> $args => ilp [1, 2, 4, 8, 16, 32] );
    };
    ( $benchmark:ident ::< $t:ty > $args:tt => ilp [ $($ilp:literal),* ] ) => {
        $(
            for_each_ilp!( $benchmark::<$t, $ilp> $args );
        )*
    };
    ( $benchmark:ident ::< $t:ty, $ilp:literal >( $($arg:expr),* ) ) => {
        if $ilp <= MIN_FLOAT_REGISTERS {
            $benchmark::<$t, $ilp>( $($arg),* );
        }
    };
}

/// Benchmark a set of ILP configurations for a given scalar/SIMD type, using
/// input from CPU registers
#[cfg(feature = "register_data_sources")]
macro_rules! for_each_registers_and_ilp {
    ( $benchmark:ident ::< $t:ty >() => (criterion $criterion:expr, tname $tname:expr) ) => {
        #[cfg(feature = "tiny_inputs")]
        for_each_registers_and_ilp!(
            $benchmark::<$t>() => (criterion $criterion, tname $tname, inputregs [1, 2, 4, 8, 16])
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
            // inputs and temporaries like square roots.
            //
            // 1-2 temporaries is enough for all the computations we're doing
            // here, therefore, configurations with <4 inputs underuse registers
            // on all known CPUs. We thus don't compile them in by default to
            // save compilation time.
            //
            // Similarly, >=32 input registers make no sense because the largest
            // platforms have 32 registers and we need some for accumulators
            $benchmark::<$t>() => (criterion $criterion, tname $tname, inputregs [4, 8, 16])
        );
    };
    ( $benchmark:ident ::< $t:ty >() => (criterion $criterion:expr, tname $tname:expr, inputregs [ $($inputregs:literal),* ]) ) => {
        $({
            let data_source = if $inputregs == 1 {
                "1register".to_string()
            } else {
                format!("{}registers", $inputregs)
            };
            let mut group = $criterion.benchmark_group(format!("{}/{data_source}", $tname));
            for_each_registers_and_ilp!(
                // We need 1 register per accumulator and current hardware has
                // at most 32 scalar/SIMD registers, so it does not make sense
                // to compile for ILP >= 32 at this point in time as we'd have
                // no registers left for input.
                $benchmark::<$t, $inputregs>(&mut group) => ilp [1, 2, 4, 8, 16]
            );
        })*
    };
    ( $benchmark:ident ::< $t:ty, $inputregs:literal >( $group:expr ) => ilp [ $($ilp:literal),* ] ) => {
        $(
            for_each_registers_and_ilp!( $benchmark::<$t, $inputregs, $ilp>($group) );
        )*
    };
    ( $benchmark:ident ::< $t:ty, $inputregs:literal, $ilp:literal >( $group:expr ) ) => {
        // Minimal register footprint = $inputregs shared inputs + $ilp accumulators
        if ($ilp + $inputregs) <= MIN_FLOAT_REGISTERS {
            $benchmark::<$t, $inputregs, $ilp>( $group );
        }
    };
}

/// Benchmark all ILP configurations for a given scalar/SIMD type
#[inline(never)] // Trying to make perf profiles look nicer
fn benchmark_type<T: Input>(c: &mut Criterion, tname: &str, interesting_data_sizes: &[usize]) {
    // Benchmark with input data that fits in CPU registers
    #[cfg(feature = "register_data_sources")]
    for_each_registers_and_ilp!(benchmark_ilp_registers::<T>() => (criterion c, tname tname));

    // Benchmark with input data that fits in L1, L2, ... all the way to RAM
    for (idx, &data_size) in interesting_data_sizes.iter().enumerate() {
        // Set up criterion for this dataset configuration
        let data_source = if cfg!(feature = "more_memory_data_sources") {
            if idx < interesting_data_sizes.len() - 1 {
                format!("L{}cache", idx + 1)
            } else {
                "RAM".to_string()
            }
        } else {
            assert_eq!(interesting_data_sizes.len(), 1);
            "L1cache".to_string()
        };
        let mut group = c.benchmark_group(format!("{tname}/{data_source}"));
        let num_elems = data_size / std::mem::size_of::<T>();

        // Allocate required storage
        let mut input_storage = vec![T::default(); num_elems];

        // Run the benchmarks at each supported ILP level
        for_each_ilp!(benchmark_ilp_memory::<T>(&mut group, &mut input_storage));
    }
}

/// Benchmark all scalar or SIMD configurations for a floating-point type, using
/// inputs from CPU registers
#[cfg(feature = "register_data_sources")]
#[inline(never)] // Trying to make perf profiles look nicer
fn benchmark_ilp_registers<T: Input, const INPUT_REGISTERS: usize, const ILP: usize>(
    group: &mut BenchmarkGroup<WallTime>,
) {
    // Iterate over subnormal configurations
    let num_subnormal_configurations = INPUT_REGISTERS.min(MAX_SUBNORMAL_CONFIGURATIONS);
    for subnormal_share in 0..=num_subnormal_configurations {
        // Generate input data
        let num_subnormals = subnormal_share * INPUT_REGISTERS / num_subnormal_configurations;
        let inputs = T::generate_positive_inputs_exact::<INPUT_REGISTERS>(num_subnormals);

        // Name this input
        let input_name = format!("{num_subnormals}in{INPUT_REGISTERS}_subnormal");

        // Run all the benchmarks on this input
        benchmark_ilp::<_, _, ILP>(
            group,
            &input_name,
            MIN_FLOAT_REGISTERS - INPUT_REGISTERS,
            inputs,
        );
    }
}

/// Benchmark all scalar or SIMD configurations for a floating-point type, using
/// inputs from memory
#[inline(never)] // Trying to make perf profiles look nicer
fn benchmark_ilp_memory<T: Input, const ILP: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    input_storage: &mut [T],
) {
    // Iterate over subnormal configurations
    for subnormal_probability in 0..=MAX_SUBNORMAL_CONFIGURATIONS {
        // Generate input data
        let subnormal_probability =
            subnormal_probability as f32 / MAX_SUBNORMAL_CONFIGURATIONS as f32;
        T::generate_positive_inputs(input_storage, subnormal_probability);

        // Name this input
        let input_name = format!("{:.0}%_subnormal", subnormal_probability * 100.0);

        // Run all the benchmarks on this input
        benchmark_ilp::<_, _, ILP>(group, &input_name, MIN_FLOAT_REGISTERS, &*input_storage);
    }
}

/// Benchmark all scalar or SIMD configurations for a floating-point type, using
/// inputs from CPU registers or memory
#[inline(never)] // Trying to make perf profiles look nicer
fn benchmark_ilp<T: Input, Inputs: InputSet<T>, const ILP: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    input_name: &str,
    processing_registers: usize,
    inputs: Inputs,
) {
    // Name this input configuration
    let ilp_name = name_ilp(ILP);
    let bench_name_prefix = format!("{ilp_name}/{input_name}");

    // Generate accumulator initial values and averaging targets
    let mut rng = rand::thread_rng();
    let normal_sampler = T::normal_sampler();
    let additive_accumulator_init =
        std::array::from_fn::<_, ILP, _>(|_| T::splat(100.0) * normal_sampler(&mut rng));
    let averaging_accumulator_init = std::array::from_fn::<_, ILP, _>(|_| normal_sampler(&mut rng));
    let average_target = normal_sampler(&mut rng);

    // Throughput in operations/second is equal to throughput in data
    // inputs/second for almost all benchmarks, except for fma_full which
    // ingests two inputs per operation.
    let num_inputs = inputs.as_ref().len();
    assert!(num_inputs >= 1);
    let num_op_data_inputs = if Inputs::MUST_SHARE {
        num_inputs * ILP // Each input is sent to each ILP stream
    } else {
        num_inputs // Each ILP stream gets its own substream of the input
    } as u64;
    group.throughput(Throughput::Elements(num_op_data_inputs));

    // Check if we should benchmark a certain ILP configuration
    let should_check_ilp = |max_ilp: usize| {
        // Round to lower power of two since we only benchmark at powers of two
        let optimal_ilp = 1 << max_ilp.ilog2();
        cfg!(feature = "more_ilp_configurations")
            || ILP == 1
            || ILP == optimal_ilp
            || ILP == optimal_ilp / 2
    };
    let max_ilp_for_binary_op = |mut remaining_registers: usize| {
        if !HAS_MEMORY_OPERANDS {
            remaining_registers /= 2
        }
        remaining_registers
    };

    // Benchmark addition and subtraction
    #[cfg(feature = "bench_addsub")]
    {
        let max_ilp = max_ilp_for_binary_op(processing_registers);
        if should_check_ilp(max_ilp) {
            group.bench_with_input(
                format!("{bench_name_prefix}/addsub"),
                &inputs,
                make_criterion_benchmark(additive_accumulator_init, addsub),
            );
        }
    }

    // Benchmark square root of positive numbers, followed by add/sub cycle
    //
    // At least one extra register is used to hold square roots before add/sub,
    // restricting available ILP for this benchmark.
    #[cfg(feature = "bench_sqrt_positive_addsub")]
    {
        let max_ilp = max_ilp_for_binary_op(processing_registers - 1);
        if should_check_ilp(max_ilp) {
            group.bench_with_input(
                format!("{bench_name_prefix}/sqrt_positive_addsub"),
                &inputs,
                make_criterion_benchmark(additive_accumulator_init, sqrt_positive_addsub),
            );
        }
    }

    // For multiplicative benchmarks, we're going to need to an extra averaging
    // operation, otherwise once we've multiplied by a subnormal we'll stay in
    // subnormal range forever, and this prevents us from studying the effect of
    // various subnormals occurence frequencies in the input stream.
    //
    // Two registers are used for the averaging (target + halving weight),
    // restricting available ILP for this benchmark.
    {
        let max_ilp = max_ilp_for_binary_op(processing_registers - 2);
        if should_check_ilp(max_ilp) {
            // First benchmark the averaging in isolation
            #[cfg(feature = "bench_average")]
            group.bench_with_input(
                format!("{bench_name_prefix}/average"),
                &inputs,
                make_criterion_benchmark(averaging_accumulator_init, average),
            );

            // Benchmark multiplication followed by averaging
            #[cfg(feature = "bench_mul_average")]
            group.bench_with_input(
                format!("{bench_name_prefix}/mul_average"),
                &inputs,
                make_criterion_benchmark(
                    averaging_accumulator_init,
                    #[inline(always)]
                    move |acc, inputs| mul_average(acc, average_target, inputs),
                ),
            );

            // Benchmark fma then averaging with possibly subnormal multiplier
            #[cfg(feature = "bench_fma_multiplier_average")]
            group.bench_with_input(
                format!("{bench_name_prefix}/fma_multiplier_average"),
                &inputs,
                make_criterion_benchmark(
                    averaging_accumulator_init,
                    #[inline(always)]
                    move |acc, inputs| fma_multiplier_average(acc, average_target, inputs),
                ),
            );

            // Benchmark fma then averaging with possibly subnormal addend
            #[cfg(feature = "bench_fma_addend_average")]
            group.bench_with_input(
                format!("{bench_name_prefix}/fma_addend_average"),
                &inputs,
                make_criterion_benchmark(
                    averaging_accumulator_init,
                    #[inline(always)]
                    move |acc, inputs| fma_addend_average(acc, average_target, inputs),
                ),
            );
        }
    }

    // Benchmark fma then averaging with possibly subnormal multiplier _and_
    // addend. This benchmark suffers from more register pressure than others
    // because FMA can only take 1 memory operand, therefore it is available in
    // even less ILP configurations.
    #[cfg(feature = "bench_fma_full_average")]
    {
        let mut max_ilp = processing_registers - 2;
        max_ilp /= if HAS_MEMORY_OPERANDS { 2 } else { 3 };
        if num_inputs >= 2 && should_check_ilp(max_ilp) {
            // Benchmark fma then averaging with possibly subnormal operands
            group.throughput(Throughput::Elements(num_op_data_inputs / 2));
            group.bench_with_input(
                format!("{bench_name_prefix}/fma_full_average"),
                &inputs,
                make_criterion_benchmark(
                    averaging_accumulator_init,
                    #[inline(always)]
                    move |acc, inputs| fma_full_average(acc, average_target, inputs),
                ),
            );
        }
    }
}

/// Name a given ILP configuration
fn name_ilp(ilp: usize) -> String {
    if ilp == 1 {
        "no_ilp".to_string()
    } else {
        format!("ilp{ilp}")
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
                let mut accumulators = accumulator_init.hide();
                let start = Instant::now();
                for _ in 0..iters {
                    let local_inputs = inputs.hide();
                    iteration(&mut accumulators, local_inputs);
                }
                let duration = start.elapsed();
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
fn sqrt_positive_addsub<T: Input, const ILP: usize>(
    accumulators: &mut [T; ILP],
    inputs: impl InputSet<T>,
) {
    iter_halves(
        accumulators,
        inputs,
        |acc, elem| *acc += elem.sqrt(),
        |acc, elem| *acc -= elem.sqrt(),
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
    let inputs = inputs.as_ref();
    let (factor_inputs, addend_inputs) = inputs.split_at(inputs.len() / 2);
    let iter = |acc: &mut T, factor, addend| {
        *acc = (acc.mul_add(factor, addend) + target) * T::splat(0.5);
    };
    if Inputs::MUST_SHARE {
        for (&factor, &addend) in factor_inputs.iter().zip(addend_inputs) {
            for acc in accumulators.iter_mut() {
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
                .zip(accumulators.iter_mut())
            {
                iter(acc, factor, addend);
            }
        }
        for ((&factor, &addend), acc) in factor_remainder
            .iter()
            .zip(addend_remainder)
            .zip(accumulators.iter_mut())
        {
            iter(acc, factor, addend);
        }
    }
}

/// Benchmark skeleton that processes the full input identically
#[allow(unused)]
#[inline(always)]
fn iter_full<T: Input, Inputs: InputSet<T>, const ILP: usize>(
    accumulators: &mut [T; ILP],
    inputs: Inputs,
    mut iter: impl FnMut(&mut T, T),
) {
    let inputs = inputs.as_ref();
    if Inputs::MUST_SHARE {
        for &elem in inputs {
            for acc in accumulators.iter_mut() {
                iter(acc, elem);
            }
        }
    } else {
        let chunks = inputs.chunks_exact(ILP);
        let remainder = chunks.remainder();
        for chunk in chunks {
            for (&elem, acc) in chunk.iter().zip(accumulators.iter_mut()) {
                iter(acc, elem);
            }
        }
        for (&elem, acc) in remainder.iter().zip(accumulators.iter_mut()) {
            iter(acc, elem);
        }
    }
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
    let inputs = inputs.as_ref();
    let (low_inputs, high_inputs) = inputs.split_at(inputs.len() / 2);
    if Inputs::MUST_SHARE {
        for (&low_elem, &high_elem) in low_inputs.iter().zip(high_inputs) {
            for acc in accumulators.iter_mut() {
                low_iter(acc, low_elem);
            }
            for acc in accumulators.iter_mut() {
                high_iter(acc, high_elem);
            }
        }
    } else {
        let low_chunks = low_inputs.chunks_exact(ILP);
        let high_chunks = high_inputs.chunks_exact(ILP);
        let low_remainder = low_chunks.remainder();
        let high_remainder = high_chunks.remainder();
        for (low_chunk, high_chunk) in low_chunks.zip(high_chunks) {
            for (&low_elem, acc) in low_chunk.iter().zip(accumulators.iter_mut()) {
                low_iter(acc, low_elem);
            }
            for (&high_elem, acc) in high_chunk.iter().zip(accumulators.iter_mut()) {
                high_iter(acc, high_elem);
            }
        }
        for (&low_elem, acc) in low_remainder.iter().zip(accumulators.iter_mut()) {
            low_iter(acc, low_elem);
        }
        for (&high_elem, acc) in high_remainder.iter().zip(accumulators.iter_mut()) {
            high_iter(acc, high_elem);
        }
    }
}

// --- Data abstraction layer ---

/// Array/slice abstraction layer
trait InputSet<T: Input>: AsRef<[T]> + Borrow<[T]> + Copy {
    /// Make a copy that is unrelated in the eyes of the optimizer
    fn hide(self) -> Self;

    /// Truth that we must use the same input for each accumulator
    const MUST_SHARE: bool;
}
//
impl<T: Input, const N: usize> InputSet<T> for [T; N] {
    #[inline(always)]
    fn hide(self) -> Self {
        self.map(pessimize::hide)
    }

    const MUST_SHARE: bool = true;
}
//
impl<T: Input> InputSet<T> for &[T] {
    #[inline(always)]
    fn hide(self) -> Self {
        pessimize::hide(self)
    }

    const MUST_SHARE: bool = false;
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
    // Given a random distribution of random numbers (distributed between 0.5
    // and 2.0 to reduce risk of overflow, underflow, etc) and the random
    // distribution of all subnormal numbers for a type, set up a random
    // positive number generator that yields a certain amount of subnormals.
    fn normal_sampler() -> impl Fn(&mut ThreadRng) -> Self;
    fn subnormal_sampler() -> impl Fn(&mut ThreadRng) -> Self;
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

    // We're also gonna need some float ops not exposed via standard traits
    fn splat(x: f32) -> Self;
    fn mul_add(self, factor: Self, addend: Self) -> Self;
    fn sqrt(self) -> Self;
}
//
impl Input for f32 {
    fn normal_sampler() -> impl Fn(&mut ThreadRng) -> Self {
        let dist = Uniform::new(0.5, 2.0);
        move |rng| rng.sample(dist)
    }

    fn subnormal_sampler() -> impl Fn(&mut ThreadRng) -> Self {
        let dist = Uniform::new(2.0f32.powi(-149), 2.0f32.powi(-126));
        move |rng| rng.sample(dist)
    }

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
        let dist = Uniform::new(0.5, 2.0);
        move |rng| rng.sample(dist)
    }

    fn subnormal_sampler() -> impl Fn(&mut ThreadRng) -> Self {
        let dist = Uniform::new(2.0f64.powi(-1074), 2.0f64.powi(-1022));
        move |rng| rng.sample(dist)
    }

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
