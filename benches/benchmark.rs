#![feature(portable_simd)]

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
use hwlocality::Topology;
use pessimize::Pessimize;
use rand::{distributions::Uniform, prelude::*};
use std::{
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    simd::{LaneCount, Simd, StdFloat, SupportedLaneCount},
};
use target_features::Architecture;

// --- Benchmark configuration and combinatorics ---

/// Maximum granularity of subnormal occurence probabilities
///
/// Higher is more precise, but the benchmark measurement time in the default
/// configuration is multipled by this factor
const MAX_SUBNORMAL_CONFIGURATIONS: usize = 2;

pub fn criterion_benchmark(c: &mut Criterion) {
    // Probe how much data that fits in L1, L2, ... and add a size that only
    // fits in RAM for completeness
    let cache_stats = Topology::new().unwrap().cpu_cache_stats().unwrap();
    let smallest_data_cache_sizes = cache_stats.smallest_data_cache_sizes();
    let interesting_data_sizes = smallest_data_cache_sizes
        .iter()
        .copied()
        .map(|dcache_size| dcache_size / 2)
        .chain(std::iter::once(
            smallest_data_cache_sizes.last().unwrap() * 8,
        ))
        .map(|size| usize::try_from(size).unwrap())
        .collect::<Vec<_>>();

    // Run benchmark for all supported data types
    benchmark_type::<f32>(c, "f32", &interesting_data_sizes);
    benchmark_type::<f64>(c, "f64", &interesting_data_sizes);
    #[cfg(feature = "types_simd")]
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
        if $ilp <= NUM_SIMD_REGISTERS {
            $benchmark::<$t, $ilp>( $($arg),* );
        }
    };
}

/// Benchmark a set of ILP configurations for a given scalar/SIMD type, using
/// input from CPU registers
macro_rules! for_each_registers_and_ilp {
    ( $benchmark:ident ::< $t:ty >() => (criterion $criterion:expr, tname $tname:expr) ) => {
        #[cfg(feature = "allow_tiny_inputs")]
        for_each_registers_and_ilp!(
            $benchmark::<$t>() => (criterion $criterion, tname $tname, inputregs [1, 2, 4, 8, 16, 32])
        );
        #[cfg(not(feature = "allow_tiny_inputs"))]
        for_each_registers_and_ilp!(
            // We need 2 regs per ILP lane (initial accumulator + current accumulator
            // and the highest ILP we target is 8. This ILP only works for platforms
            // with 32 registers, as it leaves no register left for input on platforms
            // with 16 registers. On those, the highest we can get is ILP4, which
            // consumes 8 registers for accumulators and leaves 8 free for data.
            //
            // Therefore, configurations with <8 inputs underuse CPU registers on all known
            // CPU platforms and are thus not very interesting to compile in.
            $benchmark::<$t>() => (criterion $criterion, tname $tname, inputregs [8, 16, 32])
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
                // We need 2 regs per ILP lane (initial accumulator + current accumulator)
                // and current hardware has at most 32 SIMD registers, so it does not
                // make sense to compile for more than ILP8 at this point in time
                $benchmark::<$t, $inputregs>(&mut group) => ilp [1, 2, 4, 8]
            );
        })*
    };
    ( $benchmark:ident ::< $t:ty, $inputregs:literal >( $group:expr ) => ilp [ $($ilp:literal),* ] ) => {
        $(
            for_each_registers_and_ilp!( $benchmark::<$t, $inputregs, $ilp>($group) );
        )*
    };
    ( $benchmark:ident ::< $t:ty, $inputregs:literal, $ilp:literal >( $group:expr ) ) => {
        // Minimal register footprint = $ilp initial accumulator values + $inputregs common inputs + $ilp running accumulators
        if (2 * $ilp + $inputregs) <= NUM_SIMD_REGISTERS {
            $benchmark::<$t, $inputregs, $ilp>( $group );
        }
    };
}

/// Benchmark all ILP configurations for a given scalar/SIMD type
fn benchmark_type<T: FloatLike>(c: &mut Criterion, tname: &str, interesting_data_sizes: &[usize]) {
    // Benchmark with input data that fits in CPU registers
    for_each_registers_and_ilp!(benchmark_ilp_registers::<T>() => (criterion c, tname tname));

    // Benchmark with input data that fits in L1, L2, ... all the way to RAM
    for (idx, &data_size) in interesting_data_sizes.iter().enumerate() {
        // Set up criterion for this dataset configuration
        let data_source = if idx < interesting_data_sizes.len() - 1 {
            format!("L{}cache", idx + 1)
        } else {
            "RAM".to_string()
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
fn benchmark_ilp_registers<T: FloatLike, const INPUT_REGISTERS: usize, const ILP: usize>(
    group: &mut BenchmarkGroup<WallTime>,
) {
    // Iterate over subnormal configurations
    let num_subnormal_configurations = INPUT_REGISTERS.min(MAX_SUBNORMAL_CONFIGURATIONS);
    for subnormal_share in 0..=num_subnormal_configurations {
        // Generate input data
        let num_subnormals = subnormal_share * INPUT_REGISTERS / num_subnormal_configurations;
        let inputs = T::generate_positive_inputs_exact::<INPUT_REGISTERS>(num_subnormals);

        // Run all the benchmarks on this input
        benchmark_ilp::<T, ILP, INPUT_REGISTERS>(
            group,
            num_subnormals,
            INPUT_REGISTERS,
            NUM_SIMD_REGISTERS - INPUT_REGISTERS - ILP,
            inputs,
        );
    }
}

/// Benchmark all scalar or SIMD configurations for a floating-point type, using
/// inputs from memory
fn benchmark_ilp_memory<T: FloatLike, const ILP: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    input_storage: &mut [T],
) {
    // Iterate over subnormal configurations
    for subnormal_probability in 0..=MAX_SUBNORMAL_CONFIGURATIONS {
        // Generate input data
        T::generate_positive_inputs(
            input_storage,
            subnormal_probability as f32 / MAX_SUBNORMAL_CONFIGURATIONS as f32,
        );

        // Run all the benchmarks on this input
        benchmark_ilp::<T, ILP, 0>(
            group,
            subnormal_probability,
            MAX_SUBNORMAL_CONFIGURATIONS,
            NUM_SIMD_REGISTERS,
            &*input_storage,
        );
    }
}

/// Benchmark all scalar or SIMD configurations for a floating-point type, using
/// inputs from CPU registers or memory
#[inline(never)] // Trying to make perf profiles look nicer
fn benchmark_ilp<T: FloatLike, const ILP: usize, const INPUT_REGISTERS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    subnormal_numerator: usize,
    subnormal_denominator: usize,
    processing_registers: usize,
    inputs: impl AsRef<[T]> + Copy,
) {
    // Name this input configuration
    let ilp_name = name_ilp(ILP);
    let input_name = name_subnormal(subnormal_numerator, subnormal_denominator);
    let bench_name_prefix = format!("{ilp_name}/{input_name}");

    // Generate accumulator initial values and averaging targets
    let mut rng = rand::thread_rng();
    let normal_sampler = T::normal_sampler();
    let additive_accumulator_init =
        std::array::from_fn::<_, ILP, _>(|_| T::splat(100.0) * normal_sampler(&mut rng));
    let averaging_accumulator_init = std::array::from_fn::<_, ILP, _>(|_| normal_sampler(&mut rng));
    let average_target = normal_sampler(&mut rng);

    // Throughput in operations/second is equal to throughput in data inputs/s
    // for almost all benchmarks, except for fma_full which ingests two inputs
    // per operation.
    let num_true_inputs = inputs.as_ref().len();
    let num_op_data_inputs = if INPUT_REGISTERS > 0 {
        INPUT_REGISTERS * ILP
    } else {
        num_true_inputs
    } as u64;
    group.throughput(Throughput::Elements(num_op_data_inputs));

    // Benchmark addition and subtraction
    #[cfg(feature = "bench_addsub")]
    group.bench_with_input(
        format!("{bench_name_prefix}/addsub"),
        &(additive_accumulator_init, inputs),
        |b, &(accumulator_init, inputs)| {
            b.iter(
                #[inline(always)]
                move || addsub::<T, ILP, INPUT_REGISTERS>(accumulator_init, inputs),
            )
        },
    );

    // Benchmark square root of positive numbers, followed by add/sub cycle
    //
    // Square roots of negative numbers may or may not be emulated in
    // software (due to errno shenanigans) and are thus not a good candidate
    // for CPU microbenchmarking.
    //
    // This means that the addend that we use to introduce a dependency
    // chain is always positive, so we need to flip between addition and
    // subtraction to avoid unbounded growth.
    #[cfg(feature = "bench_sqrt_positive_addsub")]
    group.bench_with_input(
        format!("{bench_name_prefix}/sqrt_positive_addsub"),
        &(additive_accumulator_init, inputs),
        |b, &(accumulator_init, inputs)| {
            b.iter(
                #[inline(always)]
                move || sqrt_positive_addsub::<T, ILP, INPUT_REGISTERS>(accumulator_init, inputs),
            )
        },
    );

    // For multiplicative benchmarks, we're going to need to an extra
    // averaging operation, otherwise once we've multiplied by a subnormal
    // we'll stay in subnormal range forever.
    //
    // Two registers are used for averaging, restricting available ILP
    let processing_registers = processing_registers - 2;
    if ILP < processing_registers {
        // First benchmark the averaging in isolation
        #[cfg(feature = "bench_average")]
        group.bench_with_input(
            format!("{bench_name_prefix}/average"),
            &(averaging_accumulator_init, inputs),
            |b, &(accumulator_init, inputs)| {
                b.iter(
                    #[inline(always)]
                    move || average::<T, ILP, INPUT_REGISTERS>(accumulator_init, inputs),
                )
            },
        );

        // Benchmark multiplication followed by averaging
        #[cfg(feature = "bench_mul_average")]
        group.bench_with_input(
            format!("{bench_name_prefix}/mul_average"),
            &(averaging_accumulator_init, inputs),
            |b, &(accumulator_init, inputs)| {
                b.iter(
                    #[inline(always)]
                    move || {
                        mul_average::<T, ILP, INPUT_REGISTERS>(
                            accumulator_init,
                            average_target,
                            inputs,
                        )
                    },
                )
            },
        );

        // Benchmark fma then averaging with possibly subnormal multiplier
        #[cfg(feature = "bench_fma_multiplier_average")]
        group.bench_with_input(
            format!("{bench_name_prefix}/fma_multiplier_average"),
            &(averaging_accumulator_init, inputs),
            |b, &(accumulator_init, inputs)| {
                b.iter(
                    #[inline(always)]
                    move || {
                        fma_multiplier_average::<T, ILP, INPUT_REGISTERS>(
                            accumulator_init,
                            average_target,
                            inputs,
                        )
                    },
                )
            },
        );

        // Benchmark fma then averaging with possibly subnormal addend
        #[cfg(feature = "bench_fma_addend_average")]
        group.bench_with_input(
            format!("{bench_name_prefix}/fma_addend_average"),
            &(averaging_accumulator_init, inputs),
            |b, &(accumulator_init, inputs)| {
                b.iter(
                    #[inline(always)]
                    move || {
                        fma_addend_average::<T, ILP, INPUT_REGISTERS>(
                            accumulator_init,
                            average_target,
                            inputs,
                        )
                    },
                )
            },
        );

        // Benchmark fma then averaging with possibly subnormal multiplier
        // _and_ addend. This benchmark has twice the register pressure, so
        // it is available in less ILP configurations.
        #[cfg(feature = "bench_fma_full_average")]
        if num_true_inputs >= 2 && ILP * 2 < processing_registers {
            // Benchmark fma then averaging with possibly subnormal multiplier
            group.throughput(Throughput::Elements(num_op_data_inputs / 2));
            group.bench_with_input(
                format!("{bench_name_prefix}/fma_full_average"),
                &(averaging_accumulator_init, inputs),
                |b, &(accumulator_init, inputs)| {
                    b.iter(
                        #[inline(always)]
                        move || {
                            fma_full_average::<T, ILP, INPUT_REGISTERS>(
                                accumulator_init,
                                average_target,
                                inputs,
                            )
                        },
                    )
                },
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

/// Name a given subnormal configuration
fn name_subnormal(numerator: usize, denominator: usize) -> String {
    if numerator == 0 {
        "no_subnormal".to_string()
    } else if 2 * numerator == denominator {
        "half_subnormal".to_string()
    } else if numerator == denominator {
        "all_subnormal".to_string()
    } else {
        format!("{numerator}in{denominator}_subnormal")
    }
}

// --- Actual benchmarks ---

/// Benchmark addition and subtraction
///
/// This is just a random additive walk of ~unity or subnormal step, so
/// given a high enough starting point, our accumulator should stay in
/// the normal range forever.
#[cfg(feature = "bench_addsub")]
#[inline(always)]
fn addsub<T: FloatLike, const ILP: usize, const INPUT_REGISTERS: usize>(
    accumulator_init: [T; ILP],
    inputs: impl AsRef<[T]>,
) {
    iter_halves::<T, ILP, INPUT_REGISTERS>(
        accumulator_init,
        inputs,
        |acc, elem| *acc += elem,
        |acc, elem| *acc -= elem,
    );
}

/// Benchmark square root of positive numbers, followed by add/sub cycle
///
/// Square roots of negative numbers may or may not be emulated in software (due
/// to errno shenanigans) and are thus not a good candidate for CPU
/// microbenchmarking.
#[cfg(feature = "bench_sqrt_positive_addsub")]
#[inline(always)]
fn sqrt_positive_addsub<T: FloatLike, const ILP: usize, const INPUT_REGISTERS: usize>(
    accumulator_init: [T; ILP],
    inputs: impl AsRef<[T]>,
) {
    iter_halves::<T, ILP, INPUT_REGISTERS>(
        accumulator_init,
        inputs,
        |acc, elem| *acc += elem.sqrt(),
        |acc, elem| *acc -= elem.sqrt(),
    );
}

/// For multiplicative benchmarks, we're going to need to an extra
/// averaging operation, otherwise once we've multiplied by a subnormal
/// we'll stay in subnormal range forever.
///
/// Averaging uses 2 CPU registers, restricting available ILP
#[cfg(feature = "bench_average")]
#[inline(always)]
fn average<T: FloatLike, const ILP: usize, const INPUT_REGISTERS: usize>(
    accumulator_init: [T; ILP],
    inputs: impl AsRef<[T]>,
) {
    iter_full::<T, ILP, INPUT_REGISTERS>(accumulator_init, inputs, |acc, elem| {
        *acc = (*acc + elem) * T::splat(0.5)
    });
}

/// Benchmark multiplication followed by averaging
#[cfg(feature = "bench_mul_average")]
#[inline(always)]
fn mul_average<T: FloatLike, const ILP: usize, const INPUT_REGISTERS: usize>(
    accumulator_init: [T; ILP],
    target: T,
    inputs: impl AsRef<[T]>,
) {
    iter_full::<T, ILP, INPUT_REGISTERS>(accumulator_init, inputs, move |acc, elem| {
        *acc = ((*acc * elem) + target) * T::splat(0.5)
    });
}

/// Benchmark fma then averaging with possibly subnormal multiplier
#[cfg(feature = "bench_fma_multiplier_average")]
#[inline(always)]
fn fma_multiplier_average<T: FloatLike, const ILP: usize, const INPUT_REGISTERS: usize>(
    accumulator_init: [T; ILP],
    target: T,
    inputs: impl AsRef<[T]>,
) {
    let halve_weight = T::splat(0.5);
    iter_full::<T, ILP, INPUT_REGISTERS>(accumulator_init, inputs, move |acc, elem| {
        *acc = (acc.mul_add(elem, halve_weight) + target) * halve_weight;
    });
}

/// Benchmark fma then averaging with possibly subnormal addend
#[cfg(feature = "bench_fma_addend_average")]
#[inline(always)]
fn fma_addend_average<T: FloatLike, const ILP: usize, const INPUT_REGISTERS: usize>(
    accumulator_init: [T; ILP],
    target: T,
    inputs: impl AsRef<[T]>,
) {
    let halve_weight = T::splat(0.5);
    iter_full::<T, ILP, INPUT_REGISTERS>(accumulator_init, inputs, move |acc, elem| {
        *acc = (acc.mul_add(halve_weight, elem) + target) * halve_weight;
    });
}

/// Benchmark fma then averaging with possibly subnormal multiplier _and_
/// addend. This benchmark has twice the register pressure (need registers for
/// both the accumulator and one of the operands), so it is available in less
/// ILP configurations.
#[cfg(feature = "bench_fma_full_average")]
#[inline(always)]
fn fma_full_average<T: FloatLike, const ILP: usize, const INPUT_REGISTERS: usize>(
    accumulator_init: [T; ILP],
    target: T,
    inputs: impl AsRef<[T]>,
) {
    let mut accumulators = accumulator_init.map(|init| pessimize::hide(init));
    let iter = |acc: &mut T, factor, addend| {
        *acc = (acc.mul_add(factor, addend) + target) * T::splat(0.5);
    };
    if INPUT_REGISTERS > 0 {
        assert_eq!(INPUT_REGISTERS % 2, 0);
        assert_eq!(
            std::mem::size_of_val(&inputs),
            INPUT_REGISTERS * std::mem::size_of::<T>()
        );
        let inputs = inputs.as_ref();
        assert_eq!(INPUT_REGISTERS, inputs.len());
        let inputs = std::array::from_fn::<T, INPUT_REGISTERS, _>(|i| inputs[i]);
        let (factor_inputs, addend_inputs) = inputs.split_at(inputs.len() / 2);
        for (&factor, &addend) in factor_inputs.iter().zip(addend_inputs) {
            for acc in &mut accumulators {
                iter(acc, factor, addend);
            }
        }
    } else {
        let inputs = inputs.as_ref();
        let (factor_inputs, addend_inputs) = inputs.split_at(inputs.len() / 2);
        let factor_chunks = factor_inputs.chunks_exact(ILP);
        let addend_chunks = addend_inputs.chunks_exact(ILP);
        let factor_remainder = factor_chunks.remainder();
        let addend_remainder = addend_chunks.remainder();
        for (factor_chunk, addend_chunk) in factor_chunks.zip(addend_chunks) {
            for ((&factor, &addend), acc) in
                factor_chunk.iter().zip(addend_chunk).zip(&mut accumulators)
            {
                iter(acc, factor, addend);
            }
        }
        for ((&factor, &addend), acc) in factor_remainder
            .iter()
            .zip(addend_remainder)
            .zip(&mut accumulators)
        {
            iter(acc, factor, addend);
        }
    }
    for acc in accumulators {
        pessimize::consume(acc);
    }
}

/// Benchmark skeleton that processes the full input symmetrically
#[allow(unused)]
#[inline(always)]
fn iter_full<T: FloatLike, const ILP: usize, const INPUT_REGISTERS: usize>(
    accumulator_init: [T; ILP],
    inputs: impl AsRef<[T]>,
    mut iter: impl FnMut(&mut T, T),
) {
    let mut accumulators = accumulator_init.map(|init| pessimize::hide(init));
    if INPUT_REGISTERS > 0 {
        assert_eq!(
            std::mem::size_of_val(&inputs),
            INPUT_REGISTERS * std::mem::size_of::<T>()
        );
        let inputs = inputs.as_ref();
        assert_eq!(INPUT_REGISTERS, inputs.len());
        let inputs = std::array::from_fn::<T, INPUT_REGISTERS, _>(|i| inputs[i]);
        for elem in inputs {
            for acc in &mut accumulators {
                iter(acc, elem);
            }
        }
    } else {
        let chunks = inputs.as_ref().chunks_exact(ILP);
        let remainder = chunks.remainder();
        for chunk in chunks {
            for (&elem, acc) in chunk.iter().zip(&mut accumulators) {
                iter(acc, elem);
            }
        }
        for (&elem, acc) in remainder.iter().zip(&mut accumulators) {
            iter(acc, elem);
        }
    }
    for acc in accumulators {
        pessimize::consume(acc);
    }
}

/// Benchmark skeleton that treats halves of the input in an asymmetric fashion
#[allow(unused)]
#[inline(always)]
fn iter_halves<T: FloatLike, const ILP: usize, const INPUT_REGISTERS: usize>(
    accumulator_init: [T; ILP],
    inputs: impl AsRef<[T]>,
    mut low_iter: impl FnMut(&mut T, T),
    mut high_iter: impl FnMut(&mut T, T),
) {
    let mut accumulators = accumulator_init.map(|init| pessimize::hide(init));
    if INPUT_REGISTERS > 0 {
        assert_eq!(INPUT_REGISTERS % 2, 0);
        assert_eq!(
            std::mem::size_of_val(&inputs),
            INPUT_REGISTERS * std::mem::size_of::<T>()
        );
        let inputs = inputs.as_ref();
        assert_eq!(INPUT_REGISTERS, inputs.len());
        let inputs = std::array::from_fn::<T, INPUT_REGISTERS, _>(|i| inputs[i]);
        let (low_inputs, high_inputs) = inputs.split_at(inputs.len() / 2);
        for (&low_elem, &high_elem) in low_inputs.iter().zip(high_inputs) {
            for acc in &mut accumulators {
                low_iter(acc, low_elem);
            }
            for acc in &mut accumulators {
                high_iter(acc, high_elem);
            }
        }
    } else {
        let inputs = inputs.as_ref();
        let (low_inputs, high_inputs) = inputs.split_at(inputs.len() / 2);
        let low_chunks = low_inputs.chunks_exact(ILP);
        let high_chunks = high_inputs.chunks_exact(ILP);
        let low_remainder = low_chunks.remainder();
        let high_remainder = high_chunks.remainder();
        for (low_chunk, high_chunk) in low_chunks.zip(high_chunks) {
            for (&low_elem, acc) in low_chunk.iter().zip(&mut accumulators) {
                low_iter(acc, low_elem);
            }
            for (&high_elem, acc) in high_chunk.iter().zip(&mut accumulators) {
                high_iter(acc, high_elem);
            }
        }
        for (&low_elem, acc) in low_remainder.iter().zip(&mut accumulators) {
            low_iter(acc, low_elem);
        }
        for (&high_elem, acc) in high_remainder.iter().zip(&mut accumulators) {
            high_iter(acc, high_elem);
        }
    }
    for acc in accumulators {
        pessimize::consume(acc);
    }
}

// --- f32/f64 and Scalar/SIMD abstraction layer ---

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
impl FloatLike for f32 {
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
impl FloatLike for f64 {
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

// --- Microarchitectural info ---

/// How many SIMD registers we can use before spilling
const NUM_SIMD_REGISTERS: usize = const {
    let target = target_features::CURRENT_TARGET;
    match target.architecture() {
        Architecture::Arm | Architecture::AArch64 => {
            if target.supports_feature_str("sve") {
                32
            } else {
                // Yes, I know, it's a little bit more complicated than that on
                // some older chips, but if you target an old enough VFP to
                // care, you probably can't run this benchmark on it anyway...
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

// --- Criterion boilerplate ---

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
