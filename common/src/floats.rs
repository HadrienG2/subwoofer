//! Abstraction over the various available floating-point data types

use pessimize::Pessimize;
use rand::{distributions::Uniform, prelude::*};
#[cfg(feature = "simd")]
use std::simd::{cmp::SimdPartialOrd, LaneCount, Simd, StdFloat, SupportedLaneCount};
use std::{
    num::NonZeroUsize,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Floating-point type that can be natively processed by the CPU
pub trait FloatLike:
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
    /// Our normal inputs follow the distribution of `2.0.pow(u)`, where `u`
    /// follows a uniform distribution from -1 to 1. We use this distribution
    /// because it has several good properties:
    ///
    /// - The numbers are close to 1, which is the optimum range for
    ///   floating-point arithmetic:
    ///     * Starting from a value of order of magnitude
    ///       `2.0.powi(MANTISSA_DIGITS/2)`, we can randomly add and subtract
    ///       numbers close to 1 for a long time before we run into significant
    ///       precision issues (due to the accumulator becoming too large) or
    ///       cancelation/underflow issues (due to the accumulator becoming too
    ///       small)
    ///     * Starting from a value of order of magnitude 1, we can multiply or
    ///       divide by values close to 1 for a long time before hitting
    ///       exponent overflow or underflow issues.
    /// - The particular distribution chosen between 0.5 to 2.0 additionally
    ///   ensures that repeatedly multiplying by numbers from this distribution
    ///   results in a random walk: an accumulator that is repeatedly multiplied
    ///   by such values should oscillate around its initial value in
    ///   multiplicative steps of at most `* 2.0` or `/ 2.0` per iteration, with
    ///   low odds of getting too large or too small if the RNG is working
    ///   correctly. If the accumulator starts close to 1, we are well protected
    ///   from exponent overflow and underflow during this random walk.
    ///
    /// For SIMD types, we generate a vector of such floats.
    ///
    /// This should be `#[inline]` because some benchmarks call this as part of
    /// their initialization procedure.
    fn normal_sampler<R: Rng>() -> impl Fn(&mut R) -> Self;

    /// Random distribution of positive subnormal floats
    ///
    /// These values are effectively treated as zeros by all of our current
    /// benchmarks, therefore their exact distribution does not matter at this
    /// point in time. We should just strive to evenly cover the space of all
    /// possible subnormals, i.e. generate a subnormal value with uniformly
    /// distributed random mantissa bits.
    ///
    /// For SIMD types, we generate a vector of such floats.
    fn subnormal_sampler<R: Rng>() -> impl Fn(&mut R) -> Self;

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

    // We're also gonna need some float data & ops not exposed via std traits.
    const MANTISSA_DIGITS: u32;
    const MIN_POSITIVE: Self;
    const MAX: Self;
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
    fn normal_sampler<R: Rng>() -> impl Fn(&mut R) -> Self {
        let dist = Uniform::new(-1.0, 1.0);
        #[inline]
        move |rng| 2.0f32.powf(rng.sample(dist))
    }

    fn subnormal_sampler<R: Rng>() -> impl Fn(&mut R) -> Self {
        let dist = Uniform::new(2.0f32.powi(-149), 2.0f32.powi(-126));
        #[inline]
        move |rng| rng.sample(dist)
    }

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

    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;
    const MAX: Self = f32::MAX;

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
    fn normal_sampler<R: Rng>() -> impl Fn(&mut R) -> Self {
        let dist = Uniform::new(-1.0, 1.0);
        #[inline]
        move |rng| 2.0f64.powf(rng.sample(dist))
    }

    fn subnormal_sampler<R: Rng>() -> impl Fn(&mut R) -> Self {
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

    const MANTISSA_DIGITS: u32 = f64::MANTISSA_DIGITS;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;
    const MAX: Self = f64::MAX;

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
    fn normal_sampler<R: Rng>() -> impl Fn(&mut R) -> Self {
        let sampler = f32::normal_sampler();
        #[inline]
        move |rng| std::array::from_fn(|_| sampler(rng)).into()
    }

    fn subnormal_sampler<R: Rng>() -> impl Fn(&mut R) -> Self {
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

    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;
    const MIN_POSITIVE: Self = Self::from_array([f32::MIN_POSITIVE; WIDTH]);
    const MAX: Self = Self::from_array([f32::MAX; WIDTH]);

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
    fn normal_sampler<R: Rng>() -> impl Fn(&mut R) -> Self {
        let sampler = f64::normal_sampler();
        #[inline]
        move |rng| std::array::from_fn(|_| sampler(rng)).into()
    }

    fn subnormal_sampler<R: Rng>() -> impl Fn(&mut R) -> Self {
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

    const MANTISSA_DIGITS: u32 = f64::MANTISSA_DIGITS;
    const MIN_POSITIVE: Self = Self::from_array([f64::MIN_POSITIVE; WIDTH]);
    const MAX: Self = Self::from_array([f64::MAX; WIDTH]);

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
