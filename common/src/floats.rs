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
    T::sampler::<R>((T::FINITE_EXPS.start + 1)..T::FINITE_EXPS.end)
}

/// Random distribution with all numbers in range [0.5; 2[
///
/// This is the basic distribution that we use when we want a tight exponent
/// range, but still coverage of all possible mantissa patterns and
/// positive/negative exponents.
#[inline]
pub fn narrow_sampler<T: FloatLike, R: Rng>() -> impl Fn(&mut R) -> T {
    T::sampler::<R>(-1..1)
}

/// Random distribution that can yield all subnormal numbers, but also has a
/// small probability of yielding zero (1 / number of fraction bits of T)
#[inline]
pub fn subnormal_zero_sampler<T: FloatLike, R: Rng>() -> impl Fn(&mut R) -> T {
    T::sampler::<R>(T::FINITE_EXPS.start..(T::FINITE_EXPS.start + 1))
}

/// Random distribution of all possible subnormal numbers
///
/// Unlike `subnormal_zero_sampler()`, this will never yield zero, but may run
/// slower as a result.
#[inline]
pub fn subnormal_sampler<T: FloatLike, R: Rng>() -> impl Fn(&mut R) -> T {
    let subnormal_or_zero = subnormal_zero_sampler::<T, R>();
    let zero = T::splat(0.0);
    move |rng: &mut R| loop {
        let result = subnormal_or_zero(rng);
        if result != zero {
            return result;
        }
    }
}
