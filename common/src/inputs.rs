//! Benchmark input datasets

use crate::floats::FloatLike;
use rand::prelude::*;

/// Owned or borrowed ordered set of benchmark inputs
///
/// We run benchmarks at all input sizes that are meaningful in hardware, from
/// inputs that can stay resident in a few CPU registers to arbitrarily large
/// in-memory datasets. But our benchmarking procedures need a few
/// size-dependent adaptations.
///
/// This trait abstracts over the two kind of inputs, and allows us to
/// manipulate them using an interface that is as homogeneous as possible.
pub trait Inputs: AsRef<[Self::Element]> {
    /// Floating-point input type
    type Element: FloatLike;

    /// Kind of input dataset, used for type-dependent computations
    const KIND: InputKind;

    /// Make the compiler think that this is an entirely different input, with a
    /// minimal impact on CPU register locality
    ///
    /// Implementations must be marked `#[inline]` to work as expected.
    fn hide_inplace(&mut self);

    /// Copy this data if owned, otherwise reborrow it as immutable
    ///
    /// This operation is needed in order to minimize the harmful side-effects
    /// of the `hide_inplace()` optimization barrier.
    ///
    /// It should be marked `#[inline]` so the compiler gets a better picture of
    /// the underlying benchmark data flow.
    fn freeze(&mut self) -> Self::Frozen<'_>;

    /// Clone of Self if possible, otherwise in-place reborrow
    type Frozen<'parent>: Inputs<Element = Self::Element>
    where
        Self: 'parent;
}
//
/// Like [`Inputs`] but allows unrestricted element mutation
///
/// Types implement [`Inputs`] but not [`InputsMut`] when they have some
/// internal invariant that they need to preserve.
pub trait InputsMut: Inputs + AsMut<[Self::Element]> {}

/// Kind of [`InputStorage`] that we are dealing with
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputKind {
    /// In-register dataset
    ///
    /// This input dataset is designed to be permanently held into the specified
    /// amount of CPU registers throughout the entire duration of a benchmark.
    ///
    /// This extra register pressure reduces the number of CPU registers that
    /// can be used as accumulators, and thus the degree of Instruction-Level
    /// Parallelism that can be used in the benchmark.
    ///
    /// To reduce this effect, inputs from an in-register dataset should be
    /// reused, i.e. each benchmark accumulator should be fed with all inputs
    /// from the dataset. This comes at the expense of reduced realism, and an
    /// increased potential for compiler over-optimization that may require
    /// stronger optimization barriers to mitigate.
    ReusedRegisters { count: usize },

    /// In-memory dataset
    ///
    /// This input dataset is designed to stay resident in one of the layers of
    /// the CPU's memory hierarchy, maybe a data cache, maybe only RAM.
    ///
    /// The associated storage is not owned by a particular benchmark, but
    /// borrowed from a longer-lived allocation.
    Memory,
}
//
impl InputKind {
    /// Truth that this kind of inputs is reused, i.e. each input from the
    /// dataset gets fed into each benchmark accumulator
    ///
    /// Otherwise, each benchmark accumulator gets fed with its own subset of
    /// the input data set.
    pub const fn is_reused(self) -> bool {
        match self {
            Self::ReusedRegisters { .. } => true,
            Self::Memory => false,
        }
    }
}

/// Total number of inputs that get aggregated into a benchmark's accumulators
///
/// This accounts for the reuse of small in-register input datasets across all
/// benchmark accumulators.
pub fn accumulated_len<I: Inputs>(inputs: &I, ilp: usize) -> usize {
    let mut result = inputs.as_ref().len();
    if I::KIND.is_reused() {
        result *= ilp;
    }
    result
}

// Implementations of InputStorage
impl<T: FloatLike, const N: usize> Inputs for [T; N] {
    type Element = T;

    const KIND: InputKind = InputKind::ReusedRegisters { count: N };

    #[inline]
    fn hide_inplace(&mut self) {
        for elem in self {
            let old_elem = *elem;
            let new_elem = pessimize::hide::<T>(old_elem);
            *elem = new_elem;
        }
    }

    fn freeze(&mut self) -> Self::Frozen<'_> {
        *self
    }

    type Frozen<'parent>
        = [T; N]
    where
        T: 'parent;
}
//
impl<T: FloatLike, const N: usize> InputsMut for [T; N] {}
//
impl<'buffer, T: FloatLike> Inputs for &'buffer mut [T] {
    type Element = T;

    const KIND: InputKind = InputKind::Memory;

    #[inline]
    fn hide_inplace<'hidden>(&'hidden mut self) {
        *self =
            // SAFETY: Although the borrow checker does not know it,
            //         pessimize::hide(*self) is just *self, so this is a no-op
            //         *self = *self instruction that is safe by definition.
            unsafe { std::mem::transmute::<&'hidden mut [T], &'buffer mut [T]>(pessimize::hide(*self)) }
    }

    fn freeze(&mut self) -> Self::Frozen<'_> {
        &mut *self
    }

    type Frozen<'parent>
        = &'parent mut [T]
    where
        Self: 'parent;
}
//
impl<T: FloatLike> InputsMut for &mut [T] {}

/// Specialized [`InputStorage`] wrapper for `addsub`-like benchmarks
///
/// The `addsub` benchmark works by splitting the input dataset in two halves,
/// then iterating over pairs of elements from both halves, adding one element
/// to the accumulator then subtracting the other element.
///
/// It is closely related to the `fma_multiplier_bidi` benchmark, which works
/// almost identically except all input values are multiplied by a constant
/// coefficient before being added to the accumulator or subtracted from it.
///
/// In those benchmarks, we can preserve the accumulator value over an
/// infinitely large number of iterations by imposing the following:
///
/// - The accumulator has a non-pathological initial value, not so big as to
///   cause overflow when adding an input and not so small as to cause underflow
///   when subtracting an input.
/// - Either the number of values in the input data stream is even, or the extra
///   element in one of the input data stream halves is set to zero so that it
///   does not affect the accumulator it is summed into.
/// - Ignoring the possibly zeroed element, the resulting equal-length halves of
///   the input data stream are identical.
///   * This entails that the number of subnormal elements in the input data
///     stream must be forced even, so that we can equally spread subnormal
///     elements on both sides of the input data stream.
///
/// This [`InputStorage`] wrapper enforces these properties at input generation
/// time, and lets you shuffle its elements in a manner which preserves them.
pub struct AddSubInputs<Storage: Inputs> {
    /// Inner data store
    storage: Storage,

    /// Truth that `generate()` has been called at least once
    generated: bool,
}
//
impl<Storage: InputsMut> AddSubInputs<Storage> {
    /// Wrap an [`InputStorage`] impl for use in an `addsub`-like benchmark
    pub fn new(storage: Storage) -> Self {
        Self {
            storage,
            generated: false,
        }
    }

    /// Fill inner dataset with about `approx_subnormals` subnormal elements
    pub fn generate(&mut self, rng: &mut impl Rng, approx_subnormals: usize) {
        // Validate input arguments
        assert!(approx_subnormals <= self.storage.as_ref().len());

        // Perform left/right/extra split, zero out extra on first run
        let generated = self.generated;
        let (left, right, extra) = self.split_storage();
        if !generated {
            if let Some(extra) = extra {
                *extra = Storage::Element::splat(0.0);
            }
        }

        // Fill the left half with the desired share of normals/subnormals...
        generate_mixture_ordered(left, rng, approx_subnormals);

        // ...and fill the right half with identical values
        right.copy_from_slice(left);

        // Record that the inner storage now contains valid inputs...
        self.generated = true;

        // ...and apply the same random permutation to both halves
        self.shuffle(rng);
    }

    /// Shuffle inner dataset in a manner that preserves the desired properties
    pub fn shuffle(&mut self, rng: &mut impl Rng) {
        assert!(self.generated);
        let (left, right, _extra) = self.split_storage();
        assert_eq!(left.len(), right.len());
        for dest_idx in (1..left.len()).rev() {
            // Apply the Durstenfeld variant of the Fisher-Yates shuffle, using
            // the same permutation for the left and right halves of the dataset
            let source_idx = rng.gen_range(0..=dest_idx);
            left.swap(source_idx, dest_idx);
            right.swap(source_idx, dest_idx);
        }
    }

    /// Split the dataset in two equally sized halves + maybe one extra element
    #[allow(clippy::type_complexity)]
    fn split_storage(
        &mut self,
    ) -> (
        &mut [Storage::Element],
        &mut [Storage::Element],
        Option<&mut Storage::Element>,
    ) {
        let storage = self.storage.as_mut();
        let half_len = storage.len() / 2;
        let (left, right) = storage.split_at_mut(half_len);
        if left.len() == right.len() {
            (left, right, None)
        } else {
            assert_eq!(right.len(), left.len() + 1);
            let (rightmost, other_right) = right.split_last_mut().unwrap();
            (left, other_right, Some(rightmost))
        }
    }
}
//
impl<Storage: Inputs> AsRef<[Storage::Element]> for AddSubInputs<Storage> {
    fn as_ref(&self) -> &[Storage::Element] {
        self.storage.as_ref()
    }
}
//
impl<Storage: Inputs> Inputs for AddSubInputs<Storage> {
    type Element = Storage::Element;

    const KIND: InputKind = Storage::KIND;

    #[inline]
    fn hide_inplace(&mut self) {
        self.storage.hide_inplace()
    }

    fn freeze(&mut self) -> Self::Frozen<'_> {
        AddSubInputs {
            storage: self.storage.freeze(),
            generated: self.generated,
        }
    }

    type Frozen<'parent>
        = AddSubInputs<Storage::Frozen<'parent>>
    where
        Self: 'parent;
}
//
impl<Storage: InputsMut> From<Storage> for AddSubInputs<Storage> {
    fn from(storage: Storage) -> Self {
        Self::new(storage)
    }
}

/// Fill up a slice with a certain share of normal and subnormal numbers
pub fn generate_mixture<T: FloatLike>(target: &mut [T], rng: &mut impl Rng, num_subnormals: usize) {
    generate_mixture_ordered(target, rng, num_subnormals);
    target.shuffle(rng);
}

/// Like `generated_mixture`, but does not randomize the location of normal and
/// subnormal values in the input.
///
/// Such a shuffling step is needed before the resulting slice is suitable for
/// use as the input to a benchmark. Yet this lower-level function is sometimes
/// needed because some benchmarks need data to be shuffled in a way that is
/// more elaborate than `target.shuffle(rng)`.
pub fn generate_mixture_ordered<T: FloatLike>(
    target: &mut [T],
    rng: &mut impl Rng,
    num_subnormals: usize,
) {
    assert!(num_subnormals <= target.len());
    let (subnormals, normals) = target.split_at_mut(num_subnormals);
    let normal = T::normal_sampler();
    let subnormal = T::subnormal_sampler();
    for elem in subnormals {
        *elem = subnormal(rng);
    }
    for elem in normals {
        *elem = normal(rng);
    }
}
