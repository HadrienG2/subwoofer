//! Abstraction over the various available floating-point data types

use pessimize::Pessimize;
use rand::prelude::*;
#[cfg(feature = "simd")]
use std::simd::{cmp::SimdPartialOrd, LaneCount, Simd, StdFloat, SupportedLaneCount};
use std::{
    fmt::Debug,
    num::NonZeroUsize,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Range, Sub, SubAssign},
};

/// Floating-point type that can be natively processed by the CPU
pub trait FloatLike:
    Add<Output = Self>
    + AddAssign
    + Copy
    + Debug
    + Default
    + Div<Output = Self>
    + DivAssign
    + Mul<Output = Self>
    + MulAssign
    + Neg<Output = Self>
    + PartialEq
    + Pessimize
    + Sub<Output = Self>
    + SubAssign
    + 'static
{
    /// Random distribution of positive IEEE-754 floats with a given exponent
    /// range and uniformly random fraction bits
    ///
    /// For example, if the provided exponent range is `0..1`, this will
    /// generate any of the floating-point numbers in range `[1.0; 2.0[` with
    /// equal probability.
    ///
    /// The specified exponent range should be a subset of the valid exponent
    /// range `FINITE_EXP_RANGE.start..(FINITE_EXP_RANGE.end+1)`.
    ///
    /// Use the exponent range
    /// `FINITE_EXP_RANGE.start..(FINITE_EXP_RANGE.start+1)` to generate mostly
    /// subnormal numbers, but also occasionally zeros with a very low
    /// probability of `2^-MANTISSA_BITS`.
    ///
    /// For SIMD types, we generate a vector of such floats.
    ///
    /// This should be `#[inline]` because the resulting RNG is critical to
    /// benchmark performance when generating large input datasets.
    fn sampler<R: Rng>(exp_range: Range<i32>) -> impl Fn(&mut R) -> Self;

    /// Range of exponents that corresponds to finite (non-NaN/non-inf) numbers
    const FINITE_EXPS: Range<i32>;

    /// Minimal number of accumulators of this type that the compiler needs to
    /// autovectorize the accumulation.
    ///
    /// `None` means that this accumulator type has the maximal supported width
    /// for this hardware. Therefore autovectorization is impossible and
    /// [`pessimize::hide()`] barriers on accumulators can be omitted.
    ///
    /// If we don't know for the active hardware, we return `Some(2)` as a safe
    /// default, which means that all but one accumulator will need to go
    /// through a `pessimize::hide()` optimization barrier.
    ///
    /// This is a workaround for rustc/LLVM spilling accumulators to memory when
    /// they are passed through [`pessimize::hide()`] in benchmarks that
    /// operates from memory inputs. See
    /// [`operations::hide_accumulators()`](crate::operations::hide_accumulators())
    /// for more context.
    const MIN_VECTORIZABLE_ILP: Option<NonZeroUsize>;

    /// Element-wise (for SIMD) minimum function that is only IEEE-754 compliant
    /// for normal, subnormal, +0 and +/-inf.
    ///
    /// If one of the inputs is a NaN or -0, the result is unspecified. This
    /// allows the use of fast hardware comparison instructions that do not have
    /// IEEE-compliant semantics.
    ///
    /// Implementations must be marked `#[inline]` as they will be called within
    /// the timed benchmark loop.
    fn fast_min(self, other: Self) -> Self;

    /// Element-wise (for SIMD) maximum function that is only IEEE-754 compliant
    /// for normal, subnormal, +0 and +/-inf.
    ///
    /// If one of the inputs is a NaN or -0, the result is unspecified. This
    /// allows the use of fast hardware comparison instructions that do not have
    /// IEEE-compliant semantics.
    ///
    /// Implementations must be marked `#[inline]` as they will be called within
    /// the timed benchmark loop.
    fn fast_max(self, other: Self) -> Self;

    // We're also gonna need some float data & ops not exposed via std traits
    fn is_normal(self) -> bool;
    fn is_subnormal(self) -> bool;
    //
    // Implementations of all of these functions must be marked `#[inline]` as
    // they will be called within the timed benchmark loop.
    fn splat(x: f32) -> Self;
    fn mul_add(self, factor: Self, addend: Self) -> Self;
    fn sqrt(self) -> Self;
}
//
impl FloatLike for f32 {
    #[inline]
    fn sampler<R: Rng>(exp_range: Range<i32>) -> impl Fn(&mut R) -> Self {
        // Check that the specified exponent range is valid
        assert!(exp_range.start >= Self::FINITE_EXPS.start);
        assert!(exp_range.start < exp_range.end);
        assert!(exp_range.end <= Self::FINITE_EXPS.end + 1);
        // From the binary32 wikipedia page
        // https://en.wikipedia.org/wiki/Single-precision_floating-point_format
        // we gather that an f32 has 23 low-order fraction bits.
        let fraction_bits = 23;
        let fraction_mask = (1 << fraction_bits) - 1;
        // Then we translate our exponent range into an exponent bits range...
        let exp_to_bits = |exp| ((exp - Self::FINITE_EXPS.start) as u32) << fraction_bits;
        let exp_bits_range = exp_to_bits(exp_range.start)..exp_to_bits(exp_range.end);
        #[inline]
        move |rng| {
            // Generate uniformly distributed random fraction bits
            let fraction_bits = rng.gen::<u32>() & fraction_mask;
            // Generate the representation of an exponent in the right range
            let exp_bits = rng.gen_range(exp_bits_range.clone());
            // Combine them to get a float with a random mantissa and the
            // desired exponent range
            f32::from_bits(fraction_bits | exp_bits)
        }
    }

    // From the binary32 wikipedia page
    // https://en.wikipedia.org/wiki/Single-precision_floating-point_format
    const FINITE_EXPS: Range<i32> = -127..128;

    const MIN_VECTORIZABLE_ILP: Option<NonZeroUsize> = const {
        if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
            // Ignoring legacy 64-bit x86 SIMD instruction sets which are not
            // even supported by compilers anymore
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

    #[inline]
    fn fast_min(self, other: Self) -> Self {
        // This generates MINSS on x86
        // TODO: Check other CPUs, add more code paths as needed
        if self < other {
            self
        } else {
            other
        }
    }

    #[inline]
    fn fast_max(self, other: Self) -> Self {
        // This generates MAXSS on x86
        // TODO: Check other CPUs, add more code paths as needed
        if self > other {
            self
        } else {
            other
        }
    }

    fn is_normal(self) -> bool {
        f32::is_normal(self)
    }

    fn is_subnormal(self) -> bool {
        f32::is_subnormal(self)
    }

    #[inline]
    fn splat(x: f32) -> Self {
        x
    }

    #[inline]
    fn mul_add(self, factor: Self, addend: Self) -> Self {
        f32::mul_add(self, factor, addend)
    }

    #[inline]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}
//
impl FloatLike for f64 {
    #[inline]
    fn sampler<R: Rng>(exp_range: Range<i32>) -> impl Fn(&mut R) -> Self {
        // Check that the specified exponent range is valid
        assert!(exp_range.start >= Self::FINITE_EXPS.start);
        assert!(exp_range.start < exp_range.end);
        assert!(exp_range.end <= Self::FINITE_EXPS.end + 1);
        // From the binary64 wikipedia page
        // https://en.wikipedia.org/wiki/Double-precision_floating-point_format
        // we gather that an f64 has 52 low-order fraction bits...
        let fraction_bits = 52;
        let fraction_mask = (1u64 << fraction_bits) - 1;
        // Then we translate our exponent range into an exponent bits range...
        let exp_to_bits = |exp| ((exp - Self::FINITE_EXPS.start) as u64) << fraction_bits;
        let exp_bits_range = exp_to_bits(exp_range.start)..exp_to_bits(exp_range.end);
        #[inline]
        move |rng| {
            // Generate uniformly distributed random fraction bits
            let fraction_bits = rng.gen::<u64>() & fraction_mask;
            // Generate the representation of an exponent in the right range
            let exp_bits = rng.gen_range(exp_bits_range.clone());
            // Combine them to get a float with a random mantissa and the
            // desired exponent range
            f64::from_bits(fraction_bits | exp_bits)
        }
    }

    // From the binary64 wikipedia page
    // https://en.wikipedia.org/wiki/Double-precision_floating-point_format
    const FINITE_EXPS: Range<i32> = -1023..1024;

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

    #[inline]
    fn fast_min(self, other: Self) -> Self {
        // This generates MINSD on x86
        // TODO: Check other CPUs, add more code paths as needed
        if self < other {
            self
        } else {
            other
        }
    }

    #[inline]
    fn fast_max(self, other: Self) -> Self {
        // This generates MAXSD on x86
        // TODO: Check other CPUs, add more code paths as needed
        if self > other {
            self
        } else {
            other
        }
    }

    fn is_normal(self) -> bool {
        f64::is_normal(self)
    }

    fn is_subnormal(self) -> bool {
        f64::is_subnormal(self)
    }

    #[inline]
    fn splat(x: f32) -> Self {
        x as f64
    }

    #[inline]
    fn mul_add(self, factor: Self, addend: Self) -> Self {
        f64::mul_add(self, factor, addend)
    }

    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}
//
#[cfg(feature = "simd")]
impl<const WIDTH: usize> FloatLike for Simd<f32, WIDTH>
where
    LaneCount<WIDTH>: SupportedLaneCount,
    Self: Pessimize + StdFloat,
{
    #[inline]
    fn sampler<R: Rng>(exp_range: Range<i32>) -> impl Fn(&mut R) -> Self {
        let sampler = f32::sampler(exp_range);
        #[inline]
        move |rng| std::array::from_fn(|_| sampler(rng)).into()
    }

    const FINITE_EXPS: Range<i32> = f32::FINITE_EXPS;

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

    #[inline]
    fn fast_min(self, other: Self) -> Self {
        // This generates (V)MINPS on x86
        // TODO: Check other CPUs, add more code paths as needed
        self.simd_lt(other).select(self, other)
    }

    #[inline]
    fn fast_max(self, other: Self) -> Self {
        // This generates (V)MAXPS on x86
        // TODO: Check other CPUs, add more code paths as needed
        self.simd_gt(other).select(self, other)
    }

    fn is_normal(self) -> bool {
        self[0].is_normal()
    }

    fn is_subnormal(self) -> bool {
        self[0].is_subnormal()
    }

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
#[cfg(feature = "simd")]
impl<const WIDTH: usize> FloatLike for Simd<f64, WIDTH>
where
    LaneCount<WIDTH>: SupportedLaneCount,
    Self: Pessimize + StdFloat,
{
    #[inline]
    fn sampler<R: Rng>(exp_range: Range<i32>) -> impl Fn(&mut R) -> Self {
        let sampler = f64::sampler(exp_range);
        #[inline]
        move |rng| std::array::from_fn(|_| sampler(rng)).into()
    }

    const FINITE_EXPS: Range<i32> = f64::FINITE_EXPS;

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

    #[inline]
    fn fast_min(self, other: Self) -> Self {
        // This generates (V)MINPD on x86
        // TODO: Check other CPUs, add more code paths as needed
        self.simd_lt(other).select(self, other)
    }

    #[inline]
    fn fast_max(self, other: Self) -> Self {
        // This generates (V)MAXPD on x86
        // TODO: Check other CPUs, add more code paths as needed
        self.simd_gt(other).select(self, other)
    }

    fn is_normal(self) -> bool {
        self[0].is_normal()
    }

    fn is_subnormal(self) -> bool {
        self[0].is_subnormal()
    }

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

/// Random distribution of all possible normal numbers
#[inline]
pub fn normal_sampler<T: FloatLike, R: Rng>() -> impl Fn(&mut R) -> T {
    T::sampler((T::FINITE_EXPS.start + 1)..T::FINITE_EXPS.end)
}

/// Random distribution with all numbers in range `[0.5; 2[`
///
/// This is the basic distribution that we use when we want a tight exponent
/// range, but still coverage of all possible mantissa patterns and
/// positive/negative exponents.
#[inline]
pub fn narrow_sampler<T: FloatLike, R: Rng>() -> impl Fn(&mut R) -> T {
    T::sampler(-1..1)
}

/// Random distribution that can yield all subnormal numbers, but also has a
/// small probability of yielding zero (1 / number of fraction bits of T)
#[inline]
pub fn subnormal_zero_sampler<T: FloatLike, R: Rng>() -> impl Fn(&mut R) -> T {
    T::sampler(T::FINITE_EXPS.start..(T::FINITE_EXPS.start + 1))
}

/// Random distribution of all possible subnormal numbers
///
/// Unlike `subnormal_zero_sampler()`, this will never yield zero, but may run
/// slower as a result.
#[inline]
pub fn subnormal_sampler<T: FloatLike, R: Rng>() -> impl Fn(&mut R) -> T {
    let subnormal_or_zero = subnormal_zero_sampler();
    let zero = T::splat(0.0);
    move |rng: &mut R| loop {
        let result = subnormal_or_zero(rng);
        if result != zero {
            return result;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_panics;
    use num_traits::{Float, NumCast};
    use proptest::prelude::*;
    use std::{num::FpCategory, slice};

    /// Number of samples used in sampler tests
    const NUM_SAMPLES: usize = 100;

    /// Reinterpret a FloatLike as an array of simpler scalar floats
    trait FloatLikeExt: FloatLike {
        type Scalar: Debug + Float;
        fn as_scalars(&self) -> &[Self::Scalar];
    }
    //
    impl FloatLikeExt for f32 {
        type Scalar = f32;
        fn as_scalars(&self) -> &[f32] {
            slice::from_ref(self)
        }
    }
    //
    impl FloatLikeExt for f64 {
        type Scalar = f64;
        fn as_scalars(&self) -> &[f64] {
            slice::from_ref(self)
        }
    }
    //
    #[cfg(feature = "simd")]
    mod simd_as_scalars {
        use super::*;
        //
        impl<const WIDTH: usize> FloatLikeExt for Simd<f32, WIDTH>
        where
            LaneCount<WIDTH>: SupportedLaneCount,
            Self: Pessimize + StdFloat,
        {
            type Scalar = f32;
            fn as_scalars(&self) -> &[f32] {
                self.as_ref()
            }
        }
        //
        impl<const WIDTH: usize> FloatLikeExt for Simd<f64, WIDTH>
        where
            LaneCount<WIDTH>: SupportedLaneCount,
            Self: Pessimize + StdFloat,
        {
            type Scalar = f64;
            fn as_scalars(&self) -> &[f64] {
                self.as_ref()
            }
        }
    }

    /// Test properties common to all FloatLike values
    fn test_value<T: FloatLikeExt>(x: T) -> Result<(), TestCaseError> {
        // Test square root
        for (&scalar, &sqrt_scalar) in x.as_scalars().iter().zip(x.sqrt().as_scalars()) {
            match scalar.classify() {
                FpCategory::Nan => prop_assert!(sqrt_scalar.is_nan()),
                FpCategory::Infinite => prop_assert_eq!(sqrt_scalar, scalar),
                FpCategory::Zero => {
                    prop_assert_eq!(sqrt_scalar, <T::Scalar as NumCast>::from(0.0f32).unwrap())
                }
                FpCategory::Subnormal | FpCategory::Normal => {
                    if scalar.is_sign_negative() {
                        prop_assert!(sqrt_scalar.is_nan());
                    } else {
                        prop_assert_eq!(sqrt_scalar, scalar.sqrt());
                    }
                }
            }
        }
        Ok(())
    }

    /// Test properties of f32-splatting
    fn test_splat<T: FloatLikeExt>(f: f32) -> Result<(), TestCaseError> {
        let x = T::splat(f);
        // T may have a wider range than f32, which means that some subnormal
        // f32s become normal Ts. Thus the only properties we can trust are...
        // - Only subnormal f32s may become subnormal Ts
        // - All normal f32s will become normal Ts
        prop_assert!(!x.is_subnormal() || f.is_subnormal());
        prop_assert!(!f.is_normal() || x.is_normal());
        for &scalar in x.as_scalars() {
            prop_assert_eq!(<f32 as NumCast>::from(scalar), Some(f));
        }
        test_value(x)?;
        Ok(())
    }
    //
    proptest! {
        #[test]
        fn splat(f: f32) {
            test_splat::<f32>(f)?;
            test_splat::<f64>(f)?;
            #[cfg(feature = "simd")]
            {
                test_splat::<Simd<f32, 4>>(f)?;
                test_splat::<Simd<f64, 2>>(f)?;
            }
        }
    }

    /// Test basic properties common to all samplers
    fn test_sampler<T: FloatLikeExt>(
        rng: &mut impl Rng,
        exp_range: Range<i32>,
    ) -> Result<(), TestCaseError> {
        // Handle panics from invalid exponent ranges
        let invalid_range = exp_range.start < T::FINITE_EXPS.start
            || exp_range.end <= exp_range.start
            || exp_range.end > T::FINITE_EXPS.end + 1;
        let exp_range2 = exp_range.clone();
        let make_sampler = || T::sampler(exp_range2);
        if invalid_range {
            return assert_panics(make_sampler);
        }

        // Check the sampler output
        let sampler = make_sampler();
        for _ in 0..NUM_SAMPLES {
            let sample = sampler(rng);

            // Global properties that can be queried via the FloatLike trait
            if exp_range.end == T::FINITE_EXPS.end + 1 {
                // Can have NaNs and +/-inf
                if exp_range.start == T::FINITE_EXPS.end {
                    // Definitiely NaN or +/-inf
                    prop_assert!(!sample.is_subnormal());
                    prop_assert!(!sample.is_normal());
                }
            } else {
                // Neither NaN nor +/-inf
                prop_assert!(exp_range.end <= T::FINITE_EXPS.end);
                if exp_range.start == T::FINITE_EXPS.start {
                    // Can have subnormal or 0
                    if exp_range.end == T::FINITE_EXPS.start + 1 {
                        // Definitely subnormal or 0
                        prop_assert!(sample.is_subnormal());
                    }
                } else {
                    // None of NaN, +/-inf, subnormal and 0
                    prop_assert!(exp_range.start > T::FINITE_EXPS.start);
                    prop_assert!(!sample.is_subnormal());
                    prop_assert!(sample.is_normal());
                }
            }

            // Properties of individual scalar elements
            for scalar in sample.as_scalars() {
                let (mut mantissa, mantissa_exponent, sign) = scalar.integer_decode();
                prop_assert_eq!(sign, 1);
                match scalar.classify() {
                    FpCategory::Nan | FpCategory::Infinite => {
                        prop_assert_eq!(exp_range.end, T::FINITE_EXPS.end + 1);
                    }
                    FpCategory::Zero | FpCategory::Subnormal => {
                        prop_assert_eq!(exp_range.start, T::FINITE_EXPS.start);
                    }
                    FpCategory::Normal => {
                        // Convert from the (mantissa, exponent) format provided
                        // by num_traits to the (fraction, exponent) format that
                        // we use inside of this module.
                        let mut fraction_exponent = mantissa_exponent;
                        while mantissa != 1 {
                            mantissa /= 2;
                            fraction_exponent += 1;
                        }
                        prop_assert!(exp_range.contains(&(fraction_exponent as i32)));
                    }
                }
            }

            // Other properties that any FloatLike should fulfill
            test_value(sample)?;
        }
        Ok(())
    }
    //
    /// Generate an exponent range that is valid for a certain FloatLike type
    fn valid_exp_range<T: FloatLike>() -> impl Strategy<Value = Range<i32>> {
        prop_oneof![
            1 => Just(T::FINITE_EXPS.start..(T::FINITE_EXPS.start+1)),
            3 => ((T::FINITE_EXPS.start+1)..T::FINITE_EXPS.end).prop_flat_map(|left| {
                ((left + 1)..=T::FINITE_EXPS.end).prop_map(move |right| {
                    left..right
                })
            }),
            1 => Just(T::FINITE_EXPS.end..(T::FINITE_EXPS.end + 1)),
        ]
    }
    //
    /// Generate an exponent range that is likely to be valid for a certain
    /// FloatLike type, but may not be
    fn exp_range<T: FloatLike>() -> impl Strategy<Value = Range<i32>> {
        prop_oneof![
            4 => valid_exp_range::<T>(),
            1 => any::<Range<i32>>()
        ]
    }
    //
    proptest! {
        #[test]
        fn sampler_f32(exp_range in exp_range::<f32>()) {
            let mut rng = rand::thread_rng();
            test_sampler::<f32>(&mut rng, exp_range.clone())?;
            #[cfg(feature = "simd")]
            test_sampler::<Simd<f32, 4>>(&mut rng, exp_range)?;
        }

        #[test]
        fn sampler_f64(exp_range in exp_range::<f64>()) {
            let mut rng = rand::thread_rng();
            test_sampler::<f64>(&mut rng, exp_range.clone())?;
            #[cfg(feature = "simd")]
            test_sampler::<Simd<f64, 2>>(&mut rng, exp_range)?;
        }
    }

    /// Test the binary and ternary operations of two FloatLike numbers
    fn test_ops<T: FloatLikeExt>(
        rng: &mut impl Rng,
        exp_range: Range<i32>,
    ) -> Result<(), TestCaseError> {
        // Fast comparison operations
        let sampler = T::sampler(exp_range);
        let x = sampler(rng);
        let y = sampler(rng);
        let has_fast_ord =
            x.as_scalars()
                .iter()
                .chain(y.as_scalars())
                .all(|scalar| match scalar.classify() {
                    FpCategory::Nan => false,
                    FpCategory::Infinite | FpCategory::Subnormal | FpCategory::Normal => true,
                    FpCategory::Zero => scalar.is_sign_positive(),
                });
        if has_fast_ord {
            for ((&x, &y), (&fast_min, &fast_max)) in ((x.as_scalars().iter()).zip(y.as_scalars()))
                .zip((x.fast_min(y).as_scalars().iter()).zip(x.fast_max(y).as_scalars()))
            {
                prop_assert_eq!(fast_min, x.min(y));
                prop_assert_eq!(fast_max, x.max(y));
            }
        }

        // Ternary operations
        let z = sampler(rng);
        for ((&x, &y), (&z, &fma)) in ((x.as_scalars().iter()).zip(y.as_scalars()))
            .zip((z.as_scalars().iter()).zip(x.mul_add(y, z).as_scalars()))
        {
            if x.is_nan() || y.is_nan() || z.is_nan() {
                prop_assert!(fma.is_nan());
            } else {
                prop_assert_eq!(fma, x.mul_add(y, z));
            }
        }
        Ok(())
    }
    //
    proptest! {
        #[test]
        fn ops_f32(exp_range in valid_exp_range::<f32>()) {
            let mut rng = rand::thread_rng();
            test_ops::<f32>(&mut rng, exp_range.clone())?;
            #[cfg(feature = "simd")]
            test_ops::<Simd<f32, 4>>(&mut rng, exp_range)?;
        }

        #[test]
        fn ops_f64(exp_range in valid_exp_range::<f64>()) {
            let mut rng = rand::thread_rng();
            test_ops::<f64>(&mut rng, exp_range.clone())?;
            #[cfg(feature = "simd")]
            test_ops::<Simd<f64, 2>>(&mut rng, exp_range)?;
        }
    }

    /// Test standardized samplers
    fn test_standard_samplers<T: FloatLikeExt>(rng: &mut impl Rng) {
        let normal = normal_sampler::<T, _>();
        for _ in 0..NUM_SAMPLES {
            assert!(normal(rng).is_normal());
        }

        let narrow = narrow_sampler::<T, _>();
        for _ in 0..NUM_SAMPLES {
            let narrow = narrow(rng);
            for &scalar in narrow.as_scalars() {
                assert!(scalar >= <T::Scalar as NumCast>::from(0.5f32).unwrap());
                assert!(scalar < <T::Scalar as NumCast>::from(2.0f32).unwrap());
            }
        }

        let sub_zero = subnormal_zero_sampler::<T, _>();
        for _ in 0..NUM_SAMPLES {
            assert!(sub_zero(rng).is_subnormal());
        }

        let subnormal = subnormal_sampler::<T, _>();
        for _ in 0..NUM_SAMPLES {
            let subnormal = subnormal(rng);
            assert!(subnormal.is_subnormal());
            assert_ne!(subnormal, T::splat(0.0));
        }
    }
    //
    #[test]
    fn standard_samplers() {
        let mut rng = rand::thread_rng();
        test_standard_samplers::<f32>(&mut rng);
        test_standard_samplers::<f64>(&mut rng);
        #[cfg(feature = "simd")]
        {
            test_standard_samplers::<Simd<f32, 4>>(&mut rng);
            test_standard_samplers::<Simd<f64, 2>>(&mut rng);
        }
    }
}
