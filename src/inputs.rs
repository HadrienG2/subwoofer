//! Benchmark input datasets

use crate::types::FloatLike;
use rand::prelude::*;

/// Maximum granularity of subnormal occurence probabilities
///
/// Higher is more precise, but the benchmark execution time in the default
/// configuration is multipled accordingly.
pub const MAX_SUBNORMAL_CONFIGURATIONS: usize = const {
    if cfg!(feature = "subnormal_freq_resolution_1in128") {
        128 // 0.78125% granularity
    } else if cfg!(feature = "subnormal_freq_resolution_1in64") {
        64 // 1.5625% granularity
    } else if cfg!(feature = "subnormal_freq_resolution_1in32") {
        32 // 3.125% granularity
    } else if cfg!(feature = "subnormal_freq_resolution_1in16") {
        16 // 6.25% granularity
    } else if cfg!(feature = "subnormal_freq_resolution_1in8") {
        8 // 12.5% granularity
    } else {
        4 // 25% granularity
    }
};

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
    /// Fill this set with a certain proportion of positive normal and subnormal
    /// inputs, taken from `T::normal_sampler()` and `T::subnormal_sampler()`
    fn generate_positive<R: Rng>(&mut self, rng: &mut R, num_subnormals: usize) {
        let slice = self.as_mut();
        assert!(num_subnormals <= slice.len());
        let (subnormal_target, normal_target) = slice.split_at_mut(num_subnormals);
        let subnormal = T::subnormal_sampler::<R>();
        for target in subnormal_target {
            *target = subnormal(rng);
        }
        let normal = T::normal_sampler::<R>();
        for target in normal_target {
            *target = normal(rng);
        }
    }

    /// Ordered sequence of inputs, borrowed from this set
    type Sequence<'a>: FloatSequence<T>
    where
        Self: 'a;

    /// Generate a randomly ordered sequence of inputs, hidden from the
    /// optimizer and initially resident in CPU registers.
    fn make_sequence(&mut self, rng: &mut impl Rng) -> Self::Sequence<'_>;

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
    fn make_sequence(&mut self, rng: &mut impl Rng) -> Self {
        self.shuffle(rng);
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
    fn make_sequence(&mut self, rng: &mut impl Rng) -> &[T] {
        self.shuffle(rng);
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
