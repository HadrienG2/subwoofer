#![feature(portable_simd)]

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
use hwlocality::Topology;
use pessimize::Pessimize;
use rand::{distributions::Uniform, rngs::ThreadRng, Rng};
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
const MAX_SUBNORMAL_CONFIGURATIONS: usize = 8;

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

/// Benchmark a set of ILP configurations for a given scalar/SIMD type
macro_rules! for_each_ilp {
    (($t:ty, $($arg:expr),*): $ilp:literal) => {
        if $ilp <= NUM_SIMD_REGISTERS {
            benchmark_ilp::<$t, $ilp>( $($arg),* );
        }
    };
    ($args:tt: $($ilp:literal),*) => {
        $(
            for_each_ilp!($args: $ilp);
        )*
    };
}

/// Benchmark all ILP configurations for a given scalar/SIMD type
fn benchmark_type<T: FloatLike>(c: &mut Criterion, tname: &str, interesting_data_sizes: &[usize]) {
    // TODO: Add benchmark solely operating on in-register data in addition to
    //       the one operating on in-memory data

    // Benchmark configurations that fit in L1, L2, ... all the way to RAM
    for (idx, &data_size) in interesting_data_sizes.iter().enumerate() {
        // Set up criterion for this dataset configuration
        let data_source = if idx < interesting_data_sizes.len() - 1 {
            format!("L{}", idx + 1)
        } else {
            "RAM".to_string()
        };
        let mut group = c.benchmark_group(format!("{tname}/{data_source}"));
        let num_elems = data_size / std::mem::size_of::<T>();
        group.throughput(Throughput::Elements(num_elems as u64));

        // Allocate required storage
        let mut input_storage = vec![T::default(); num_elems];

        // Run the benchmarks at each supported ILP level
        for_each_ilp!((T, &mut group, &mut input_storage): 1, 2, 4, 8, 16, 32);
    }
}

/// Benchmark all scalar or SIMD configurations for a floating-point type
fn benchmark_ilp<T: FloatLike, const ILP: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    input_storage: &mut [T],
) {
    // Name this ILP configuration
    let ilp_name = if ILP == 1 {
        "latency-bound".to_string()
    } else {
        format!("ilp{ILP}")
    };

    // Iterate over subnormal configurations
    for subnormal_probability in 0..=MAX_SUBNORMAL_CONFIGURATIONS {
        // Name this input configuration
        let input_name = if subnormal_probability == 0 {
            "all_normal".to_string()
        } else if 2 * subnormal_probability == MAX_SUBNORMAL_CONFIGURATIONS {
            "half_subnormal".to_string()
        } else if subnormal_probability == MAX_SUBNORMAL_CONFIGURATIONS {
            "all_subnormal".to_string()
        } else {
            format!("subnormal_{subnormal_probability}in{MAX_SUBNORMAL_CONFIGURATIONS}")
        };
        let bench_name_prefix = format!("{ilp_name}/{input_name}");

        // Generate input data
        T::generate_positive_inputs(
            input_storage,
            subnormal_probability as f32 / MAX_SUBNORMAL_CONFIGURATIONS as f32,
        );

        // Generate accumulator initial values and averaging targets
        let normal_sampler = T::normal_sampler();
        let mut rng = rand::thread_rng();
        let additive_accumulator_init =
            std::array::from_fn::<_, ILP, _>(|_| T::splat(100.0) * normal_sampler(&mut rng));
        let averaging_accumulator_init =
            std::array::from_fn::<_, ILP, _>(|_| normal_sampler(&mut rng));
        let average_target = normal_sampler(&mut rng);

        // Benchmark addition and subtraction
        group.bench_with_input(
            format!("{bench_name_prefix}/addsub"),
            input_storage,
            |b, inputs| b.iter(move || addsub(additive_accumulator_init, inputs)),
        );

        // For multiplicative benchmarks, we're going to need to an extra
        // averaging operation, otherwise once we've multiplied by a subnormal
        // we'll stay in subnormal range forever.
        //
        // Two registers are used for averaging, restricting available ILP
        let remaining_registers = NUM_SIMD_REGISTERS - 2;
        if ILP < remaining_registers {
            // First benchmark the averaging in isolation
            group.bench_with_input(
                format!("{bench_name_prefix}/average"),
                input_storage,
                |b, inputs| b.iter(move || average(averaging_accumulator_init, inputs)),
            );

            // Benchmark multiplication followed by averaging
            group.bench_with_input(
                format!("{bench_name_prefix}/mul_average"),
                input_storage,
                |b, inputs| {
                    b.iter(move || mul_average(averaging_accumulator_init, average_target, inputs))
                },
            );

            // Benchmark fma then averaging with possibly subnormal multiplier
            group.bench_with_input(
                format!("{bench_name_prefix}/fma_multiplier_average"),
                input_storage,
                |b, inputs| {
                    b.iter(move || {
                        fma_multiplier_average(averaging_accumulator_init, average_target, inputs)
                    })
                },
            );

            // Benchmark fma then averaging with possibly subnormal addend
            group.bench_with_input(
                format!("{bench_name_prefix}/fma_addend_average"),
                input_storage,
                |b, inputs| {
                    b.iter(move || {
                        fma_addend_average(averaging_accumulator_init, average_target, inputs)
                    })
                },
            );

            // Benchmark fma then averaging with possibly subnormal multiplier
            // _and_ addend. This benchmark has twice the register pressure, so
            // it is available in less ILP configurations.
            if ILP * 2 < remaining_registers {
                // Benchmark fma then averaging with possibly subnormal multiplier
                group.bench_with_input(
                    format!("{bench_name_prefix}/fma_full_average"),
                    input_storage,
                    |b, inputs| {
                        b.iter(move || {
                            fma_full_average(averaging_accumulator_init, average_target, inputs)
                        })
                    },
                );
            }
        }

        // Benchmark square root of positive numbers, followed by add/sub cycle
        //
        // Square roots of negative numbers may or may not be emulated in
        // software (due to errno shenanigans) and are thus not a good candidate
        // for CPU microbenchmarking.
        //
        // This means that the addend that we use to introduce a dependency
        // chain is always positive, so we need to flip between addition and
        // subtraction to avoid unbounded growth.
        group.bench_with_input(
            format!("{bench_name_prefix}/sqrt_positive_addsub"),
            input_storage,
            |b, inputs| b.iter(move || sqrt_positive_addsub(additive_accumulator_init, inputs)),
        );
    }
}

// --- Actual benchmarks ---

/// Benchmark addition and subtraction
///
/// This is just a random additive walk of ~unity or subnormal step, so
/// given a high enough starting point, our accumulator should stay in
/// the normal range forever.
fn addsub<T: FloatLike, const ILP: usize>(mut accumulators: [T; ILP], inputs: &[T]) {
    let chunks = inputs.chunks_exact(2 * ILP);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let (pos_chunk, neg_chunk) = chunk.split_at(ILP);
        for (&elem, acc) in pos_chunk.iter().zip(&mut accumulators) {
            *acc += elem;
        }
        for (&elem, acc) in neg_chunk.iter().zip(&mut accumulators) {
            *acc -= elem;
        }
    }
    for (idx, &elem) in remainder.iter().enumerate() {
        accumulators[idx % ILP] += elem;
    }
    for acc in accumulators {
        pessimize::consume(acc);
    }
}

/// For multiplicative benchmarks, we're going to need to an extra
/// averaging operation, otherwise once we've multiplied by a subnormal
/// we'll stay in subnormal range forever.
///
/// Averaging uses 2 CPU registers, restricting available ILP
fn average<T: FloatLike, const ILP: usize>(mut accumulators: [T; ILP], inputs: &[T]) {
    let chunks = inputs.chunks_exact(ILP);
    let remainder = chunks.remainder();
    let iter = |acc: &mut T, elem| *acc = (*acc + elem) * T::splat(0.5);
    for chunk in chunks {
        for (&elem, acc) in chunk.iter().zip(&mut accumulators) {
            iter(acc, elem);
        }
    }
    for (&elem, acc) in remainder.iter().zip(&mut accumulators) {
        iter(acc, elem);
    }
    for acc in accumulators {
        pessimize::consume(acc);
    }
}

/// Benchmark multiplication followed by averaging
fn mul_average<T: FloatLike, const ILP: usize>(
    mut accumulators: [T; ILP],
    target: T,
    inputs: &[T],
) {
    let chunks = inputs.chunks_exact(ILP);
    let remainder = chunks.remainder();
    let iter = |acc: &mut T, elem| *acc = ((*acc * elem) + target) * T::splat(0.5);
    for chunk in chunks {
        for (&elem, acc) in chunk.iter().zip(&mut accumulators) {
            iter(acc, elem);
        }
    }
    for (&elem, acc) in remainder.iter().zip(&mut accumulators) {
        iter(acc, elem);
    }
    for acc in accumulators {
        pessimize::consume(acc);
    }
}

/// Benchmark fma then averaging with possibly subnormal multiplier
fn fma_multiplier_average<T: FloatLike, const ILP: usize>(
    mut accumulators: [T; ILP],
    target: T,
    inputs: &[T],
) {
    let chunks = inputs.chunks_exact(ILP);
    let remainder = chunks.remainder();
    let halve_weight = T::splat(0.5);
    let iter = |acc: &mut T, elem| {
        *acc = (acc.mul_add(elem, halve_weight) + target) * halve_weight;
    };
    for chunk in chunks {
        for (&elem, acc) in chunk.iter().zip(&mut accumulators) {
            iter(acc, elem);
        }
    }
    for (&elem, acc) in remainder.iter().zip(&mut accumulators) {
        iter(acc, elem);
    }
    for acc in accumulators {
        pessimize::consume(acc);
    }
}

/// Benchmark fma then averaging with possibly subnormal addend
fn fma_addend_average<T: FloatLike, const ILP: usize>(
    mut accumulators: [T; ILP],
    target: T,
    inputs: &[T],
) {
    let chunks = inputs.chunks_exact(ILP);
    let remainder = chunks.remainder();
    let halve_weight = T::splat(0.5);
    let iter = |acc: &mut T, elem| {
        *acc = (acc.mul_add(halve_weight, elem) + target) * halve_weight;
    };
    for chunk in chunks {
        for (&elem, acc) in chunk.iter().zip(&mut accumulators) {
            iter(acc, elem);
        }
    }
    for (&elem, acc) in remainder.iter().zip(&mut accumulators) {
        iter(acc, elem);
    }
    for acc in accumulators {
        pessimize::consume(acc);
    }
}

/// Benchmark fma then averaging with possibly subnormal multiplier _and_
/// addend. This benchmark has twice the register pressure (need registers for
/// both the accumulator and one of the operands), so it is available in less
/// ILP configurations.
fn fma_full_average<T: FloatLike, const ILP: usize>(
    mut accumulators: [T; ILP],
    target: T,
    inputs: &[T],
) {
    let halve_weight = T::splat(0.5);
    let chunks = inputs.chunks_exact(2 * ILP);
    let remainder = chunks.remainder();
    let iter = |acc: &mut T, elem1, elem2| {
        *acc = (acc.mul_add(elem1, elem2) + target) * halve_weight;
    };
    for chunk in chunks {
        let (pos_chunk, neg_chunk) = chunk.split_at(ILP);
        for ((&elem1, &elem2), acc) in pos_chunk.iter().zip(neg_chunk).zip(&mut accumulators) {
            iter(acc, elem1, elem2);
        }
    }
    for (&elem, acc) in remainder.iter().zip(&mut accumulators) {
        iter(acc, elem, elem);
    }
    for acc in accumulators {
        pessimize::consume(acc);
    }
}

/// Benchmark square root of positive numbers, followed by add/sub cycle
///
/// Square roots of negative numbers may or may not be emulated in software (due
/// to errno shenanigans) and are thus not a good candidate for CPU
/// microbenchmarking.
fn sqrt_positive_addsub<T: FloatLike, const ILP: usize>(mut accumulators: [T; ILP], inputs: &[T]) {
    let chunks = inputs.chunks_exact(2 * ILP);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let (pos_chunk, neg_chunk) = chunk.split_at(ILP);
        for (&elem, acc) in pos_chunk.iter().zip(&mut accumulators) {
            *acc += elem.sqrt();
        }
        for (&elem, acc) in neg_chunk.iter().zip(&mut accumulators) {
            *acc -= elem.sqrt();
        }
    }
    for (idx, &elem) in remainder.iter().enumerate() {
        accumulators[idx % ILP] += elem.sqrt();
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
