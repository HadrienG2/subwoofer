//! Benchmark input datasets

use crate::{
    arch::MIN_FLOAT_REGISTERS,
    floats::{self, FloatLike},
};
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

/// Generate a mixture of normal and subnormal inputs for a benchmark that
/// follows the `acc -> acc + input * constant` pattern where `constant` is a
/// positive number in the usual "narrow" range `[0.5; 2[`, and it is assumed
/// that the accumulator is initially in that same range.
///
/// Overall, every time we add a normal value to the accumulator, we subtract it
/// back on the next normal input, thusly restoring the accumulator to its
/// initial value for the next iteration.
///
/// The normal inputs will be of magnitude `[0.5; 2[`, alternating between a
/// positive and negative sign, except for the last input which may be `0.0` in
/// order to guarantee that the value of the accumulator at the end of a
/// benchmark run is back to where it was at the beginning.
pub fn generate_add_inputs<Storage: InputsMut, const ILP: usize>(
    target: &mut Storage,
    rng: &mut impl Rng,
    num_subnormals: usize,
) {
    // Determine how many interleaved input data streams should be generated...
    let num_data_streams = if Storage::KIND.is_reused() { 1 } else { ILP };
    inner(target.as_mut(), rng, num_subnormals, num_data_streams);
    //
    // ...and dispatch accordingly to a less generic version of this function,
    // to reduce build time at an insignifiant runtime performance cost
    fn inner<T: FloatLike, R: Rng>(
        target: &mut [T],
        rng: &mut R,
        num_subnormals: usize,
        num_data_streams: usize,
    ) {
        // Decide if the next element of `target` should be subnormal
        let mut pick_subnormal = subnormal_picker(target.len(), num_subnormals);

        // Position and value of previous normal inputs that we need to recover
        // from by subtracting them back from the corresponding accumulator
        #[derive(Clone, Copy)]
        struct PrevNormal<T: FloatLike> {
            idx: usize,
            value: T,
        }
        let mut prev_normals = [None; MIN_FLOAT_REGISTERS];
        assert!(num_data_streams < prev_normals.len());
        let prev_normals = &mut prev_normals[..num_data_streams];

        // Generate an input element, knowing the associated PrevNormal state
        // for the corresponding accumulator
        let mut idx = 0;
        let normal = floats::narrow_sampler::<T, R>();
        let subnormal = floats::subnormal_sampler::<T, R>();
        let mut generate_element = |prev_normal: &mut Option<PrevNormal<T>>| {
            let value = if pick_subnormal(rng) {
                // Subnormals do not change the accumulator (because their
                // magnitude is so much lower) and can be ignored
                subnormal(rng)
            } else {
                match prev_normal.take() {
                    // If we previously added a value, subtract it back
                    Some(PrevNormal { idx: _, value }) => -value,

                    // Otherwise, generate a normal value and take note of the
                    // fact that we'll need to subtract it later on.
                    None => {
                        let value = normal(rng);
                        *prev_normal = Some(PrevNormal { idx, value });
                        value
                    }
                }
            };
            idx += 1;
            value
        };

        // Generate input data
        let mut target_chunks = target.chunks_exact_mut(num_data_streams);
        for chunk in target_chunks.by_ref() {
            for (target, prev_normal) in chunk.iter_mut().zip(prev_normals.iter_mut()) {
                *target = generate_element(prev_normal);
            }
        }
        for (target, prev_normal) in target_chunks
            .into_remainder()
            .iter_mut()
            .zip(prev_normals.iter_mut())
        {
            *target = generate_element(prev_normal);
        }

        // The last normal input cannot be canceled out, so zero it out instead
        let zero = T::splat(0.0);
        for prev_normal in prev_normals {
            if let Some(PrevNormal { idx, value: _ }) = *prev_normal {
                target[idx] = zero;
            }
        }
    }
}

// TODO: Implement a generate_mul_max_inputs that starts from a copy-paste of
//       generate_add_inputs, then modifies what's needed. Then extract any
//       implementation commonalities. Then do the same with
//       generate_div_denominator_min_inputs, and again extract commonalities
//       with mul_max. Finally do the same with
//       generate_div_numerator_max_inputs. Finally, extract single-benchmark
//       utilities to the crate of their respective benchmark.

/// Generate a mixture of normal and subnormal inputs for a benchmark that
/// follows the `acc -> max(acc, f(input))` pattern, where the accumulator is
/// initially a normal number and f is a function that turns any normal number
/// into another normal number.
pub fn generate_max_inputs<T: FloatLike, R: Rng>(
    target: &mut [T],
    rng: &mut R,
    num_subnormals: usize,
) {
    // Split the target slice into one part for normal numbers and another part
    // for subnormal numbers
    assert!(num_subnormals <= target.len());
    let (subnormals, normals) = target.split_at_mut(num_subnormals);

    // Generate the subnormal inputs
    let subnormal = floats::subnormal_sampler();
    for elem in subnormals {
        *elem = subnormal(rng);
    }

    // Generate the normal inputs
    let normal = floats::normal_sampler();
    for elem in normals {
        *elem = normal(rng);
    }

    // Randomize the order of normal and subnormal inputs
    target.shuffle(rng)
}

/// Random boolean distribution that iteratively tells if each element of a
/// slice of future benchmark inputs of size `target_len` should be subnormal
///
/// Should be called exactly `target_len` times. Will yield `num_subnormals`
/// times the value `true` and yield the value `false` the rest of the time.
fn subnormal_picker<R: Rng>(
    mut target_len: usize,
    mut num_subnormals: usize,
) -> impl FnMut(&mut R) -> bool {
    move |rng| {
        debug_assert!(target_len > 0);
        let subnormal_pos = rng.gen_range(0..target_len);
        let is_subnormal = subnormal_pos < num_subnormals;
        target_len -= 1;
        num_subnormals -= (is_subnormal) as usize;
        is_subnormal
    }
}
