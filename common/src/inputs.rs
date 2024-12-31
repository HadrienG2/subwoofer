//! Benchmark input datasets

use crate::floats::FloatLike;
use rand::prelude::*;

/// Unordered collection of benchmark inputs, to be ordered before use
///
/// For sufficiently small collections of inputs (register inputs being the
/// extreme case), the measured benchmark performance may depend on the position
/// of subnormal inputs in the input data sequence.
///
/// This is for example known to be the case for the `fma_full` benchmark on
/// current Intel CPUs: since the trigger for subnormal slowdown on those CPUs
/// is that at least one input to an instruction is subnormal, and two
/// subnormals inputs do not increase overhead, an input data configuration for
/// which half of the FMAs have all-normal inputs and half of the FMAs have
/// all-subnormal inputs should be twice as fast as another configuration where
/// all FMAs have one normal and one subnormal input.
///
/// Further, because the subnormal fallback of some hardware trashes the CPU
/// frontend, we could also expect a Sufficiently Weird CPU Microarchitecture to
/// have a subnormal-induced slowdown that varies depending on where in the
/// input program the affected instructions are located, i.e. how close they are
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
/// The former is achieved by [`generate_positive()`], while the latter is
/// achieved by an input reordering step between benchmark iteration batches.
/// The [`FloatSet`]/[`FloatSequence`] dichotomy enforces such a reordering
/// step.
pub trait FloatSet: AsMut<[Self::Element]> {
    /// Floating-point elements that compose this set
    type Element: FloatLike;

    /// Number of elements of this type inside of the set, accounting for
    /// accumulator reuse
    fn reused_len(&self, num_accumulators: usize) -> usize;

    /// Ordered sequence of inputs, borrowed from this set
    type Sequence<'a>: FloatSequence<Element = Self::Element>
    where
        Self: 'a;

    /// Generate a randomly ordered sequence of inputs from this set, hidden
    /// from the optimizer and initially resident in CPU registers.
    ///
    /// Implementations should be marked `#[inline]` to give the compiler more
    /// visibility of the data flow in the benchmark loop.
    fn make_sequence(&mut self, rng: &mut impl Rng) -> Self::Sequence<'_>;

    /// Re-expose useful FloatSequence properties for easier access
    const IS_REUSED: bool = Self::Sequence::<'_>::IS_REUSED;
    const NUM_REGISTER_INPUTS: Option<usize> = Self::Sequence::<'_>::NUM_REGISTER_INPUTS;
}
//
impl<T: FloatLike, const N: usize> FloatSet for [T; N] {
    type Element = T;

    #[allow(clippy::assertions_on_constants)]
    fn reused_len(&self, num_accumulators: usize) -> usize {
        assert!(<Self as FloatSet>::IS_REUSED);
        N * num_accumulators
    }

    type Sequence<'a>
        = Self
    where
        Self: 'a;

    #[inline]
    fn make_sequence(&mut self, rng: &mut impl Rng) -> Self {
        self.shuffle(rng);
        <[T; N] as FloatSequence>::hide_inplace(self);
        *self
    }
}
//
impl<T: FloatLike> FloatSet for &mut [T] {
    type Element = T;

    #[allow(clippy::assertions_on_constants)]
    fn reused_len(&self, _num_accumulators: usize) -> usize {
        assert!(!<Self as FloatSet>::IS_REUSED);
        <[T]>::len(self)
    }

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
pub trait FloatSequence: AsRef<[Self::Element]> + Copy {
    /// Floating-point elements that compose this sequence
    type Element: FloatLike;

    /// Pass each inner value through `pessimize::hide()`. This ensures that...
    ///
    /// - The returned values are unrelated to the original values in the eyes
    ///   of the optimizer. This is needed to avoids compiler over-optimization
    ///   of benchmarks that reuse a small set of inputs (e.g. hoisting of
    ///   `sqrt()` computations out of the iteration loop in `sqrt` benchmarks
    ///   with register inputs).
    /// - Each inner value ends up confined in its own CPU register. This can be
    ///   e.g. applied to the accumulator set between accumulation steps to
    ///   avoid unwanted compiler autovectorization of scalar accumulation code.
    ///
    /// Implementations must be marked `#[inline]` as this may be used inside of
    /// the timed benchmark loop.
    fn hide_inplace(&mut self);

    /// CPU floating-point registers that are used by this input data
    ///
    /// None means that the input comes from memory (CPU cache or RAM).
    const NUM_REGISTER_INPUTS: Option<usize>;

    /// Truth that we must reuse the same input for each accumulator
    const IS_REUSED: bool;
}
//
impl<T: FloatLike, const N: usize> FloatSequence for [T; N] {
    type Element = T;

    #[inline]
    fn hide_inplace(&mut self) {
        for elem in self {
            let old_elem = *elem;
            let new_elem = pessimize::hide::<T>(old_elem);
            *elem = new_elem;
        }
    }

    const NUM_REGISTER_INPUTS: Option<usize> = Some(N);

    const IS_REUSED: bool = true;
}
//
impl<T: FloatLike> FloatSequence for &[T] {
    type Element = T;

    #[inline]
    fn hide_inplace(&mut self) {
        *self = pessimize::hide::<&[T]>(*self)
    }

    const NUM_REGISTER_INPUTS: Option<usize> = None;

    const IS_REUSED: bool = false;
}
