//! Benchmark input datasets

use crate::{
    arch::MIN_FLOAT_REGISTERS,
    floats::{self, FloatLike},
};
use rand::prelude::*;
use std::marker::PhantomData;

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
/// follows the `acc -> max(acc, f(input))` pattern, where the accumulator is
/// initially a normal number and f is a function that turns any positive normal
/// number into another positive normal number.
///
/// These benchmarks can tolerate any sequence of normal and subnormal numbers,
/// without any risk of overflow or cancelation. Hence they get the simplest and
/// most general input generation procedure.
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

/// Generate a mixture of normal and subnormal inputs for a benchmark that
/// follows the `acc -> acc + input * constant` pattern where both `constant`
/// and the initial accumulator value are positive numbers in the usual "narrow"
/// range `[0.5; 2[`.
///
/// Overall, the general strategy to avoid unbounded accumulator growth or
/// cancelation is that...
///
/// - Generated normal inputs are in the same magnitude range as the initial
///   accumulator value `[0.5; 2[`
/// - Every time we add a normal value to the accumulator, we subtract it back
///   on the next normal input, thusly bringing the accumulator back to its
///   initial value (except for a few zeroed low-order mantissa bits)
/// - In case the number of normal inputs is odd, we cannot do this with the
///   last input, so we set it to `0` instead
pub fn generate_add_inputs<Storage: InputsMut, const ILP: usize>(
    target: &mut Storage,
    rng: &mut impl Rng,
    num_subnormals: usize,
) {
    /// (Lack of) global state for this input generator
    struct AddGenerator;
    //
    impl<T: FloatLike, R: Rng> Generator<T, R> for AddGenerator {
        type Stream<'a> = AddStream<T>;

        fn new_stream(&self) -> Self::Stream<'_> {
            AddStream::Unconstrained
        }
    }

    /// Per-stream state for this input generator
    enum AddStream<T: FloatLike> {
        /// No constraint on the next normal value
        Unconstrained,

        /// Next normal value should be the opposite of the previous one
        OppositeOf {
            /// Previous normal input value
            value: T,

            /// Global position of this normal value in the generated data
            global_idx: usize,
        },
    }
    //
    impl<T: FloatLike, R: Rng> StreamState<R> for AddStream<T> {
        type Float = T;

        #[inline]
        fn record_subnormal(&mut self) {}

        #[inline]
        fn generate_normal(
            &mut self,
            global_idx: usize,
            rng: &mut R,
            mut narrow: impl FnMut(&mut R) -> T,
        ) -> T {
            match *self {
                Self::Unconstrained => {
                    let value = narrow(rng);
                    *self = Self::OppositeOf { global_idx, value };
                    value
                }
                Self::OppositeOf { value, .. } => {
                    *self = Self::Unconstrained;
                    -value
                }
            }
        }

        fn finalize(self, mut stream: Stream<'_, T>) {
            if let Self::OppositeOf { global_idx, .. } = self {
                *stream.at_global_idx(global_idx) = T::splat(0.0);
            }
        }
    }

    // Generate benchmark inputs
    generate_arbitrary_inputs::<_, _, ILP>(target, rng, num_subnormals, AddGenerator);
}

// TODO: Use generate_muldiv_inputs in mul_max, div_denominator_min and
//       div_numerator_max, adjust the lower and upper bounds to 0.25 and 4.0,
//       fix the rest

/// Generate normal and subnormal inputs for a benchmark that follows one of the
/// `acc -> max(acc * input, 1/4)`, `acc -> min(acc / input, 4) ` and `acc ->
/// max(input / acc, 1/4)` patterns, where the initial accumulator value is a
/// positive number in the usual "narrow" range `[1/2; 2[`.
///
/// Overall, the general strategy to avoid unbounded accumulator growth or
/// shrinkage is that...
///
/// - Generated normal inputs are in the same magnitude range as the initial
///   accumulator value `[1/2; 2[`
/// - Whenever a normal input is followed by another normal input, that second
///   input is chosen to compensate the effect of the first one and reset the
///   underlying accumulator back to its starting point (or close to it). For
///   example, if we multiply an accumulator by a normal value and the next
///   input value is normal, we will make it the inverse of the previous value.
///   * The `invert_normal` callback indicates how, given a previous normal
///     input, one can compute the next normal input that cancels out its effect
///     on the accumulator.
///   * We cannot do this with the last input of a data stream, therefore we
///     force that input to be `1` instead (which does not modify the
///     accumulator's order of magnitude), so that the accumulator goes back to
///     its initial `[1/2; 2[` magnitude at the end of a benchmark run.
/// - Every time we integrate a sequence of one or more subnormal values, we
///   lose the original accumulator value and saturate to the lower/upper bound.
///   To compensate for this, the next normal input is chosen to bring the
///   accumulator back to its initial range. For example, if we multiply by a
///   subnormal input and saturate to `0.25`, the next input is picked in range
///   `[2; 8[` in order to bring the accumulator back to the `[1/2; 2[` range.
///   * The `cancel_subnormal` callback indicates how, given the normal input
///     that follows a subnormal value, one can grow or shrink it to cancel out
///     the accumulator-clamping effect of the previous subnormal value and
///     restore the accumulator to its initial [1/2; 2[ range.
///   * We cannot do this if the last input value of a data stream is subnormal,
///     therefore we impose that either the last input is normal (in which case
///     this problem doesn't exist) or the first input is subnormal (in which
///     case the initial accumulator value is immediately destroyed on the first
///     iteration of the next run and it doesn't matter that it has an unusual
///     magnitude). This is achieved by enforcing that whenever the first input
///     value is normal and the last input value is subnormal, they are swapped
///     and what is now the new last normal input value is regenerated.
pub fn generate_muldiv_inputs<Storage: InputsMut, const ILP: usize>(
    target: &mut Storage,
    rng: &mut impl Rng,
    num_subnormals: usize,
    invert_normal: impl Fn(Storage::Element) -> Storage::Element + 'static,
    cancel_subnormal: impl Fn(Storage::Element) -> Storage::Element + 'static,
) {
    /// Global state for this input generator
    struct MulDivGenerator<T: FloatLike, InvertNormal, CancelSubnormal>
    where
        InvertNormal: Fn(T) -> T,
        CancelSubnormal: Fn(T) -> T,
    {
        /// Compute the inverse of a normal input
        invert_normal: InvertNormal,

        /// Modify a normal input to cancel the effect of a previous subnormal
        cancel_subnormal: CancelSubnormal,

        /// Mark T as used so rustc doesn't go mad
        _unused: PhantomData<T>,
    }
    //
    impl<T: FloatLike, R: Rng, InvertNormal, CancelSubnormal> Generator<T, R>
        for MulDivGenerator<T, InvertNormal, CancelSubnormal>
    where
        InvertNormal: Fn(T) -> T + 'static,
        CancelSubnormal: Fn(T) -> T + 'static,
    {
        type Stream<'a> = MulDivStream<'a, T, InvertNormal, CancelSubnormal>;

        fn new_stream(&self) -> Self::Stream<'_> {
            MulDivStream {
                generator: self,
                state: MulDivState::Unconstrained,
            }
        }
    }

    /// Per-stream state for this input generator
    struct MulDivStream<'generator, T: FloatLike, InvertNormal, CancelSubnormal>
    where
        InvertNormal: Fn(T) -> T,
        CancelSubnormal: Fn(T) -> T,
    {
        /// Underlying generator state
        generator: &'generator MulDivGenerator<T, InvertNormal, CancelSubnormal>,

        /// Per-stream state machine
        state: MulDivState<T>,
    }
    //
    /// Inner state machine of [`MulDivStream`]
    enum MulDivState<T: FloatLike> {
        /// No constraint on the next normal value
        Unconstrained,

        /// Next normal value should cancel out the accumulator change caused by
        /// the previous normal input value
        InvertNormal(T),

        /// Next normal value should cancel out the accumulator low/high-bound
        /// clamping caused by the previous subnormal value
        CancelSubnormal,
    }
    //
    impl<T: FloatLike, R: Rng, InvertNormal, CancelSubnormal> StreamState<R>
        for MulDivStream<'_, T, InvertNormal, CancelSubnormal>
    where
        InvertNormal: Fn(T) -> T + 'static,
        CancelSubnormal: Fn(T) -> T + 'static,
    {
        type Float = T;

        #[inline]
        fn record_subnormal(&mut self) {
            self.state = MulDivState::CancelSubnormal;
        }

        #[inline]
        fn generate_normal(
            &mut self,
            _global_idx: usize,
            rng: &mut R,
            mut narrow: impl FnMut(&mut R) -> T,
        ) -> T {
            match self.state {
                MulDivState::Unconstrained => {
                    let value = narrow(rng);
                    self.state = MulDivState::InvertNormal(value);
                    value
                }
                MulDivState::InvertNormal(value) => {
                    self.state = MulDivState::Unconstrained;
                    (self.generator.invert_normal)(value)
                }
                MulDivState::CancelSubnormal => {
                    self.state = MulDivState::Unconstrained;
                    (self.generator.cancel_subnormal)(narrow(rng))
                }
            }
        }

        fn finalize(self, stream: Stream<'_, T>) {
            let mut stream = stream.into_iter();
            match self.state {
                MulDivState::Unconstrained => {}
                MulDivState::InvertNormal(_) => {
                    let last = stream
                        .last()
                        .expect("InvertNormal only reachable after >= 1 normal input");
                    debug_assert!(last.is_normal());
                    *last = T::splat(1.0);
                }
                MulDivState::CancelSubnormal => {
                    let first = stream
                        .next()
                        .expect("CancelSubnormal only reachable after >= 1 subnormal");
                    if first.is_normal() {
                        let last = stream
                            .last()
                            .expect("Last input should be subnormal, it cannot be the first input if it's normal");
                        debug_assert!(last.is_subnormal());
                        std::mem::swap(first, last);
                        // FIXME: Fix up new last value so that it makes sense
                        //        in the context of any previous normal values.
                        //        It should either be the inverse of the
                        //        previous normal value or 1.0, depending on
                        //        whether the resulting streams of normal
                        //        numbers has an even or odd length.
                        unimplemented!()
                    }
                }
            }
        }
    }

    // Generate benchmark inputs
    generate_arbitrary_inputs::<_, _, ILP>(
        target,
        rng,
        num_subnormals,
        MulDivGenerator {
            invert_normal,
            cancel_subnormal,
            _unused: PhantomData,
        },
    );
}

/// Generic input data generation procedure
///
/// This is the common logic that most `generate_xyz_inputs` functions plug
/// into, except for `generate_max_inputs` which can follow a simpler logic.
fn generate_arbitrary_inputs<Storage: InputsMut, R: Rng, const ILP: usize>(
    target: &mut Storage,
    rng: &mut R,
    num_subnormals: usize,
    generator: impl Generator<Storage::Element, R>,
) {
    // Determine how many interleaved input data streams should be generated...
    let num_streams = if Storage::KIND.is_reused() { 1 } else { ILP };
    inner(target.as_mut(), rng, num_subnormals, generator, num_streams);
    //
    // ...then dispatch to a less generic backend function to reduce code bloat
    fn inner<T: FloatLike, R: Rng, G: Generator<T, R>>(
        target: &mut [T],
        rng: &mut R,
        num_subnormals: usize,
        generator: G,
        num_streams: usize,
    ) {
        // Decide if the next element of `target` should be subnormal
        let mut pick_subnormal = subnormal_picker(target.len(), num_subnormals);

        // Set up data stream generators
        let mut stream_states: [_; MIN_FLOAT_REGISTERS] =
            std::array::from_fn(|_| generator.new_stream());
        assert!(num_streams < stream_states.len());
        let stream_states = &mut stream_states[..num_streams];

        // Recipe for the next element of a particular data stream
        let narrow = floats::narrow_sampler::<T, R>();
        let subnormal = floats::subnormal_sampler::<T, R>();
        let mut global_idx = 0;
        let mut generate_element = |state: &mut G::Stream<'_>| {
            let result = if pick_subnormal(rng) {
                state.record_subnormal();
                subnormal(rng)
            } else {
                state.generate_normal(global_idx, rng, &narrow)
            };
            global_idx += 1;
            result
        };

        // Generate each element of each data stream
        let mut target_chunks = target.chunks_exact_mut(num_streams);
        for chunk in target_chunks.by_ref() {
            for (target, state) in chunk.iter_mut().zip(stream_states.iter_mut()) {
                *target = generate_element(state);
            }
        }
        for (target, state) in target_chunks
            .into_remainder()
            .iter_mut()
            .zip(stream_states.iter_mut())
        {
            *target = generate_element(state);
        }

        // Fix up data streams at the end to enforce accumulator reset
        for (stream_idx, state) in stream_states
            .iter_mut()
            .map(|state| std::mem::replace(state, generator.new_stream()))
            .enumerate()
        {
            state.finalize(Stream {
                target,
                stream_idx,
                num_streams,
            })
        }
    }
}

/// Shared state of an input data generator
trait Generator<T: FloatLike, R: Rng> {
    /// State associated to one specific data stream of an input generator
    type Stream<'a>: StreamState<R, Float = T>
    where
        Self: 'a;

    /// Set up a new data stream
    ///
    /// Implementations should not assume anything from the number of times that
    /// this function will be called. For boring technical reasons, we may need
    /// to set up more data streams than we are actually going to use.
    fn new_stream(&self) -> Self::Stream<'_>;
}

/// Input data stream from a particular [`Generator`]
trait StreamState<R: Rng> {
    /// Type of floating-point value that is being generated
    type Float: FloatLike;

    /// Take note that a subnormal number has been generated
    fn record_subnormal(&mut self);

    /// Generate the next normal input value
    ///
    /// `global_idx` uniquely identifies the input value in the full dataset ,
    /// and can be used to easily locate it back at the
    /// `loopback_normal_values()` stage.
    ///
    /// `rng` is the random number generator to be used, and `narrow` is the
    /// usual `[1/2; 2[` random distribution that should be used whenever the
    /// target benchmark imposes no other constraint.
    fn generate_normal(
        &mut self,
        global_idx: usize,
        rng: &mut R,
        narrow: impl FnMut(&mut R) -> Self::Float,
    ) -> Self::Float;

    /// Finalize the previously generated input data stream
    ///
    /// Sometimes it is necessary to adjust the first/last few generated normal
    /// values in order to ensure that the accumulator goes back to a state with
    /// properties (e.g. magnitude) similar to those of its initial state after
    /// processing the entire data stream, and thus to avoid unbounded
    /// accumulator growth/shrinkage over many benchmark iterations.
    ///
    /// This method is called after right after all values have been generated,
    /// so it is guaranteed that the last normal value in the stream will be the
    /// last one that was generated by `generate_normal()`.
    ///
    /// Subnormal values are exposed in the stream because they can affect the
    /// procedure, but they should neither be modified nor repositioned.
    fn finalize(self, stream: Stream<'_, Self::Float>);
}

/// Single input data stream in the context of a global data set
struct Stream<'data, T: FloatLike> {
    target: &'data mut [T],
    stream_idx: usize,
    num_streams: usize,
}
//
impl<'data, T: FloatLike> Stream<'data, T> {
    /// Iterate over data from the specified data stream
    fn into_iter(self) -> impl DoubleEndedIterator<Item = &'data mut T> + ExactSizeIterator {
        debug_assert!(self.stream_idx < self.num_streams);
        self.target
            .iter_mut()
            .skip(self.stream_idx)
            .step_by(self.num_streams)
    }

    /// Access data by global position
    ///
    /// The specified global index must belong to the data stream of interest.
    /// So given that subnormal values should not be inspected, it should
    /// previously have been received by a
    /// [`generate_normal()`](StreamState::generate_normal) implementation.
    fn at_global_idx(&mut self, global_idx: usize) -> &mut T {
        debug_assert!(self.stream_idx < self.num_streams);
        assert_eq!(global_idx % self.num_streams, self.stream_idx);
        &mut self.target[global_idx]
    }
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
