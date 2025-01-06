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
    /// Unless a specific benchmark dictates otherwise, our inputs follow the
    /// distribution given by the following probabilistic experiment:
    ///
    /// - Generate a random number which is uniformly distributed in [1, 2[
    ///   (i.e. exponent is 0, each mantissa bit pattern has equal probability)
    /// - Flip a conceptual coin, decide accordingly whether to return the
    ///   previously generated number or its inverse.
    ///
    /// We favor this distribution because it has several good properties:
    ///
    /// - A fully random mantissa maximally protects against Sufficiently Weird
    ///   Hardware special-casing convenient numbers like 1 and 2.
    /// - The numbers are close to 1, in range ]1/2; 2[, which is good for all
    ///   basic floating-point arithmetic:
    ///     * Starting from a value of order of magnitude
    ///       `2.0.powi(MANTISSA_DIGITS/2)`, we can add and subtract random
    ///       numbers close to 1 for a long time before we run into significant
    ///       precision issues (due to the accumulator becoming too large) or
    ///       underflow issues (due to the accumulator becoming too small)
    ///     * Starting from a value of order of magnitude 1, we can multiply or
    ///       divide by random values close to 1 for a long time before hitting
    ///       exponent overflow or underflow issues.
    /// - The particular distribution chosen from 1/2 to 2 additionally ensures
    ///   that if we initialize an accumulator close to 1, then repeatedly
    ///   multiply or divide it by numbers from this distribution, we get a
    ///   random walk: the accumulator should oscillate around its initial value
    ///   in multiplicative steps of at most `* 2.0` or `/ 2.0` per iteration,
    ///   with low odds of exponent overflow and underflow.
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
    /// possible subnormals, i.e. generate a subnormal value where all mantissa
    /// bit patterns occur with equal probability.
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
        // From the binary32 wikipedia page
        // https://en.wikipedia.org/wiki/Single-precision_floating-point_format
        // we gather that an f32 has 23 low-order fraction bits...
        let fraction_bits = 23;
        let fraction_mask = (1 << fraction_bits) - 1;
        // ...and an exponent bias of 0x7f, which means that we must set the
        // exponent to 0x7f when generating numbers in [1; 2[. We leave the sign
        // bit at 0 since we're only generating positive numbers.
        let positive_exp0 = 0x7f << fraction_bits;
        // We will also need to probe the high-order bit of the RNG's output.
        let high_order_bit = 1 << (u32::BITS - 1);
        #[inline]
        move |rng| {
            // Generate a single random 32-bit numbers
            let random_bits = rng.gen::<u32>();
            // Use the low-order random bits as the fraction of our number...
            let fraction_bits = random_bits & fraction_mask;
            // ...which otherwise has an exponent of zero and a positive sign
            let one_to_two = f32::from_bits(positive_exp0 | fraction_bits);
            // Use the high-order random bit to decide whether we should invert
            // the outcome of this random number generation
            let should_invert = random_bits & high_order_bit != 0;
            if should_invert {
                1.0 / one_to_two
            } else {
                one_to_two
            }
        }
    }

    fn subnormal_sampler<R: Rng>() -> impl Fn(&mut R) -> Self {
        // Generate all subnormal fraction bit patterns with equal proability
        let subnormal_fraction = Uniform::from(1..2u32.pow(23));
        #[inline]
        move |rng| {
            // Subnormal positive numbers have...
            // - A sign bit equal to zero (because they are positive)
            // - An exponent equal to the minimal value (0)
            // - Fraction bits distributed as discussed above
            f32::from_bits(rng.sample(subnormal_fraction))
        }
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
        // From the binary64 wikipedia page
        // https://en.wikipedia.org/wiki/Double-precision_floating-point_format
        // we gather that an f64 has 52 low-order fraction bits...
        let fraction_bits = 52;
        let fraction_mask = (1u64 << fraction_bits) - 1;
        // ...and an exponent bias of 0x3ff, which means that we must set the
        // exponent to 0x3ff when generating numbers in [1; 2[. We leave the
        // sign bit at 0 since we're only generating positive numbers.
        let positive_exp0 = 0x3ff_u64 << fraction_bits;
        // We will also need to probe the high-order bit of the RNG's output.
        let high_order_bit = 1u64 << (u64::BITS - 1);
        #[inline]
        move |rng| {
            // Generate a single random 64-bit number
            let random_bits = rng.gen::<u64>();
            // Use the low-order random bits as the fraction of our number...
            let fraction_bits = random_bits & fraction_mask;
            // ...which otherwise has an exponent of zero and a positive sign
            let one_to_two = f64::from_bits(positive_exp0 | fraction_bits);
            // Use the high-order random bit to decide whether we should invert
            // the outcome of this random number generation
            let should_invert = random_bits & high_order_bit != 0;
            if should_invert {
                1.0 / one_to_two
            } else {
                one_to_two
            }
        }
    }

    fn subnormal_sampler<R: Rng>() -> impl Fn(&mut R) -> Self {
        // Generate all subnormal fraction bit patterns with equal proability
        let subnormal_fraction = Uniform::from(1..2u64.pow(52));
        #[inline]
        move |rng| {
            // Subnormal positive numbers have...
            // - A sign bit equal to zero (because they are positive)
            // - An exponent equal to the minimal value (0)
            // - Fraction bits distributed as discussed above
            f64::from_bits(rng.sample(subnormal_fraction))
        }
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
