//! Abstraction over the various available floating-point data types

use pessimize::Pessimize;
use rand::{distributions::Uniform, prelude::*};
use std::{
    num::NonZeroUsize,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    simd::{LaneCount, Simd, StdFloat, SupportedLaneCount},
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

    /// Minimal number of accumulators of this type that the compiler needs to
    /// autovectorize the accumulation.
    ///
    /// `None` means that this accumulator type has the maximal supported width
    /// for this hardware. Therefore autovectorization is impossible and
    /// `pessimize::hide()` barriers on accumulators can be omitted.
    ///
    /// If we don't know for the active hardware, we return `Some(2)` as a safe
    /// default, which means that all but one accumulator will need to go
    /// through a `pessimize::hide()` optimization barrier.
    ///
    /// This is a workaround for rustc/LLVM spilling accumulators to memory when
    /// they are passed through `pessimize::hide()` in benchmarks that operates
    /// from memory inputs. See [`hide_accumulators()`] for more context.
    const MIN_VECTORIZABLE_ILP: Option<NonZeroUsize>;

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

    const MIN_VECTORIZABLE_ILP: Option<NonZeroUsize> = const {
        if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
            // Ignoring MMX, 3DNow!, and other legacy 64-bit x86 SIMD
            // instruction sets which are not supported by compilers anymore
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

    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;

    #[inline]
    fn splat(x: f32) -> Self {
        x
    }

    #[inline]
    fn mul_add(self, factor: Self, addend: Self) -> Self {
        self.mul_add(factor, addend)
    }

    #[inline]
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

    const MANTISSA_DIGITS: u32 = f64::MANTISSA_DIGITS;

    #[inline]
    fn splat(x: f32) -> Self {
        x as f64
    }

    #[inline]
    fn mul_add(self, factor: Self, addend: Self) -> Self {
        self.mul_add(factor, addend)
    }

    #[inline]
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

    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;

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

    const MANTISSA_DIGITS: u32 = f64::MANTISSA_DIGITS;

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
pub trait FloatSet<T: FloatLike>: AsMut<[T]> {
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

    #[inline]
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

    #[inline]
    fn make_sequence(&mut self) -> &[T] {
        self.shuffle(&mut rand::thread_rng());
        pessimize::hide::<&[T]>(*self)
    }
}

/// Randomly ordered sequence of inputs that is ready for benchmark consumption
pub trait FloatSequence<T: FloatLike>: AsRef<[T]> + Copy {
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
    #[inline]
    fn hide(mut self) -> Self {
        for elem in self.iter_mut() {
            *elem = pessimize::hide::<T>(*elem);
        }
        self
    }

    const NUM_REGISTER_INPUTS: Option<usize> = Some(N);

    const IS_REUSED: bool = true;
}
//
impl<T: FloatLike> FloatSequence<T> for &[T] {
    #[inline]
    fn hide(self) -> Self {
        pessimize::hide::<&[T]>(self)
    }

    const NUM_REGISTER_INPUTS: Option<usize> = None;

    const IS_REUSED: bool = false;
}
