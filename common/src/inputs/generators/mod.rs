//! Benchmark input generation algorithms

pub mod add;
pub mod max;
pub mod muldiv;

use super::InputsMut;
use crate::{
    arch::MIN_FLOAT_REGISTERS,
    floats::{self, FloatLike},
};
use rand::prelude::*;

/// Generic procedure for generating a benchmark input dataset composed of N
/// interleaved data streams
///
/// This is the common logic that most `generate_xyz_inputs` functions plug
/// into, except for `generate_max_inputs` which can generate fully random data
/// and does not need to acknowledge the existence of interleaved data streams.
fn generate_input_streams<Storage: InputsMut, R: Rng, const ILP: usize>(
    target: &mut Storage,
    rng: &mut R,
    num_subnormals: usize,
    generator: impl InputGenerator<Storage::Element, R>,
) {
    // Determine how many interleaved input data streams should be generated...
    assert!(ILP > 0);
    let num_streams = if Storage::KIND.is_reused() { 1 } else { ILP };
    inner(target.as_mut(), rng, num_subnormals, generator, num_streams);
    //
    // ...then dispatch to a less generic backend to reduce code bloat
    fn inner<T: FloatLike, R: Rng, G: InputGenerator<T, R>>(
        target: &mut [T],
        rng: &mut R,
        num_subnormals: usize,
        generator: G,
        num_streams: usize,
    ) {
        // Validate configuration
        assert!(num_subnormals <= target.len());

        // Set up the interleaved data streams
        let mut streams: [_; MIN_FLOAT_REGISTERS] = std::array::from_fn(|_| generator.new_stream());
        assert!(num_streams <= streams.len());
        let streams = &mut streams[..num_streams];

        // Generate each element of each data stream
        {
            let mut generate_element = element_generator(num_subnormals, target.len(), rng);
            let mut target_chunks = target.chunks_exact_mut(num_streams);
            let mut global_idx = 0;
            for chunk in target_chunks.by_ref() {
                for (target, stream) in chunk.iter_mut().zip(streams.iter_mut()) {
                    *target = generate_element(stream, global_idx);
                    global_idx += 1;
                }
            }
            for (target, stream) in target_chunks
                .into_remainder()
                .iter_mut()
                .zip(streams.iter_mut())
            {
                *target = generate_element(stream, global_idx);
                global_idx += 1;
            }
        }

        // Tweak final data streams to enforce a perfect accumulator reset
        for (stream_idx, stream) in streams
            .iter_mut()
            .map(|stream| std::mem::replace(stream, generator.new_stream()))
            .enumerate()
        {
            stream.finalize(
                DataStream {
                    target,
                    stream_idx,
                    num_streams,
                },
                rng,
            )
        }
    }
}

/// Like [`generate_input_streams()`] but the dataset is split in two even
/// halves and each data stream is composed of pairs of elements from each half
/// (which itself contains the usual interleaved data layout).
///
/// For this to work, `target` must contain an even number of elements.
///
/// This is needed for benchmarks where each accumulator receives data from two
/// input data streams that play an asymmetric role, like the `acc <-
/// max(fma(acc, input1, input2), lower_bound)` benchmark.
pub fn generate_input_pairs<Storage: InputsMut, R: Rng, const ILP: usize>(
    target: &mut Storage,
    rng: &mut R,
    num_subnormals: usize,
    generator: impl InputGenerator<Storage::Element, R>,
) {
    // Determine how many interleaved input data streams should be generated...
    let num_streams = if Storage::KIND.is_reused() { 1 } else { ILP };
    inner(target.as_mut(), rng, num_subnormals, generator, num_streams);
    //
    // ...then dispatch to a less generic backend to reduce code bloat
    fn inner<T: FloatLike, R: Rng, G: InputGenerator<T, R>>(
        target: &mut [T],
        rng: &mut R,
        num_subnormals: usize,
        generator: G,
        num_streams: usize,
    ) {
        // Validate configuration
        assert!(num_subnormals <= target.len());

        // Set up the interleaved data streams
        let mut streams: [_; MIN_FLOAT_REGISTERS] = std::array::from_fn(|_| generator.new_stream());
        assert!(num_streams <= streams.len());
        let streams = &mut streams[..num_streams];

        // Split the input in two halves
        let target_len = target.len();
        assert_eq!(target_len % 2, 0);
        let half_len = target_len / 2;
        let (left_target, right_target) = target.split_at_mut(half_len);
        assert_eq!(left_target.len(), right_target.len());

        // Generate each element of each data stream: regular bulk where each
        // data stream gets one new element...
        {
            let mut generate_element = element_generator(num_subnormals, target_len, rng);
            let mut left_chunks = left_target.chunks_exact_mut(num_streams);
            let mut right_chunks = right_target.chunks_exact_mut(num_streams);
            let left_start = 0;
            let right_start = half_len;
            let mut offset = 0;
            'chunks: loop {
                match (left_chunks.next(), right_chunks.next()) {
                    (Some(left_chunk), Some(right_chunk)) => {
                        for ((left_elem, right_elem), stream) in left_chunk
                            .iter_mut()
                            .zip(right_chunk)
                            .zip(streams.iter_mut())
                        {
                            *left_elem = generate_element(stream, left_start + offset);
                            *right_elem = generate_element(stream, right_start + offset);
                            offset += 1;
                        }
                    }
                    (None, None) => break 'chunks,
                    (Some(_), None) | (None, Some(_)) => {
                        unreachable!("not possible with equal-length halves")
                    }
                }
            }

            // ...then irregular remainder where not all streams get a new element
            {
                let mut left_elems = left_chunks.into_remainder().iter_mut();
                let mut right_elems = right_chunks.into_remainder().iter_mut();
                let mut streams = streams.iter_mut();
                'elems: loop {
                    match ((left_elems.next(), right_elems.next()), streams.next()) {
                        ((Some(left_elem), Some(right_elem)), Some(stream)) => {
                            *left_elem = generate_element(stream, left_start + offset);
                            *right_elem = generate_element(stream, right_start + offset);
                            offset += 1;
                        }
                        ((None, None), Some(_)) => break 'elems,
                        ((Some(_), None), _) | ((None, Some(_)), _) => {
                            unreachable!("not possible with equal-length halves")
                        }
                        (_, None) => {
                            unreachable!("should have remainder.len() < num_streams")
                        }
                    }
                }
            }
        }

        // Tweak final data streams to enforce a perfect accumulator reset
        for (stream_idx, stream) in streams
            .iter_mut()
            .map(|stream| std::mem::replace(stream, generator.new_stream()))
            .enumerate()
        {
            stream.finalize(
                DataStream {
                    target,
                    stream_idx,
                    num_streams,
                },
                rng,
            )
        }
    }
}

/// Input data generator that produces multiple interleaved data stream
pub trait InputGenerator<T: FloatLike, R: Rng> {
    /// State associated to one specific data stream
    type Stream<'a>: GeneratorStream<R, Float = T>
    where
        Self: 'a;

    /// Set up a new data stream
    ///
    /// For boring technical resons, this function may be called more times than
    /// there are actual data streams in the generated dataset, and it may be
    /// even called again after the previously created data streams have been
    /// acted upon for a while. The resulting dummy streams will not be used,
    /// they just act as placeholder values inside of some internal container.
    ///
    /// Implementations of this trait should take this constraint into account:
    ///
    /// - They should refrain from interpreting the number of times this method
    ///   is called as being equal to the number of data streams in flight
    /// - They should not assume that `new_stream()` won't be called again after
    ///   some of the previously created streams start being used.
    ///
    /// If you have no other choice, it is however acceptable to spawn dummy
    /// unusable data streams that panic on use, after the expected number of
    /// data streams has been created.
    fn new_stream(&self) -> Self::Stream<'_>;
}

/// Data stream from a particular [`InputGenerator`]
pub trait GeneratorStream<R: Rng> {
    /// Type of floating-point value that is being generated
    type Float: FloatLike;

    /// Take note that a subnormal number has been generated
    fn record_subnormal(&mut self);

    /// Generate the next normal value
    ///
    /// `global_idx` uniquely identifies the generated value in the full dataset
    /// and can be used to locate it at the `loopback_normal_values()` stage.
    ///
    /// `rng` is the random number generator to be used, and `narrow` is the
    /// usual `[1/2; 2[` random distribution that should be used with this RNG
    /// whenever the target benchmark imposes no other constraint.
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
    /// This method is called right after all values have been generated, so it
    /// is guaranteed that the last normal value in the stream will be the last
    /// one that was generated by `generate_normal()`.
    ///
    /// Subnormal values are exposed in the stream because they can affect the
    /// procedure, but they should not be modified. They may, however, be
    /// repositioned in the data stream if necessary, provided that the result
    /// is subsequently fixed up to make it look as if the subnormal value was
    /// there from the beginning.
    fn finalize(self, stream: DataStream<'_, Self::Float>, rng: &mut R);
}

/// Single input data stream in the context of a global data set composed of
/// multiple interleaved data streams.
///
/// Can be interpreted either as a stream of scalar elements of type T, or pairs
/// thereof. Only one interpretation is correct for a particular data set.
pub struct DataStream<'data, T> {
    target: &'data mut [T],
    stream_idx: usize,
    num_streams: usize,
}
//
impl<'data, T> DataStream<'data, T> {
    /// Access a scalar element by global position
    ///
    /// The specified global index must belong to the data stream of interest.
    /// So given that subnormal values should not be modified, it should
    /// previously have been received by a
    /// [`generate_normal()`](GeneratorStream::generate_normal) implementation.
    pub fn scalar_at(&mut self, global_idx: usize) -> &mut T {
        debug_assert!(self.stream_idx < self.num_streams);
        &mut self.target[global_idx]
    }

    /// Inteprete this data stream as a stream of scalar data elements
    pub fn into_scalar_iter(
        self,
    ) -> impl DoubleEndedIterator<Item = &'data mut T> + ExactSizeIterator {
        debug_assert!(self.stream_idx < self.num_streams);
        self.target
            .iter_mut()
            .skip(self.stream_idx)
            .step_by(self.num_streams)
    }

    /// Inteprete this data stream as a stream of pairs of data elements
    pub fn into_pair_iter(
        self,
    ) -> impl DoubleEndedIterator<Item = [&'data mut T; 2]> + ExactSizeIterator {
        assert_eq!(self.target.len() % 2, 0);
        debug_assert!(self.stream_idx < self.num_streams);

        let half_len = self.target.len() / 2;
        let (left, right) = self.target.split_at_mut(half_len);

        left.iter_mut()
            .zip(right)
            .skip(self.stream_idx)
            .step_by(self.num_streams)
            .map(|(l, r)| [l, r])
    }
}

/// Recipe for iteratively producing substreams of a broader dataset
///
/// This produces a function that, given the state of a particular substream of
/// the dataset and the index of the next element, produces this element.
fn element_generator<R: Rng, S: GeneratorStream<R>>(
    num_subnormals: usize,
    target_len: usize,
    rng: &mut R,
) -> impl FnMut(&mut S, usize) -> S::Float + '_ {
    assert!(num_subnormals <= target_len);
    let narrow = floats::narrow_sampler::<S::Float, R>();
    let subnormal = floats::subnormal_sampler::<S::Float, R>();
    let mut pick_subnormal = subnormal_picker(num_subnormals, target_len);
    move |state: &mut S, global_idx: usize| {
        debug_assert!(global_idx < target_len);
        if pick_subnormal(rng) {
            state.record_subnormal();
            subnormal(rng)
        } else {
            state.generate_normal(global_idx, rng, &narrow)
        }
    }
}

/// Random boolean distribution that iteratively tells if each element of a
/// slice of future benchmark inputs of size `target_len` should be subnormal
///
/// Should be called exactly `target_len` times. Will yield `num_subnormals`
/// times the value `true` and yield the value `false` the rest of the time.
fn subnormal_picker<R: Rng>(
    mut num_subnormals: usize,
    mut target_len: usize,
) -> impl FnMut(&mut R) -> bool {
    assert!(num_subnormals <= target_len);
    move |rng| {
        debug_assert!(target_len > 0);
        debug_assert!(num_subnormals <= target_len);
        let subnormal_pos = rng.gen_range(0..target_len);
        let is_subnormal = subnormal_pos < num_subnormals;
        target_len -= 1;
        num_subnormals -= (is_subnormal) as usize;
        is_subnormal
    }
}

/// Test utilities that are shared by this crate and other crates in the namespace
#[cfg(any(test, feature = "unstable_test"))]
pub mod test_utils {
    use proptest::{prelude::*, sample::SizeRange};

    /// Length of a dataset that can be reinterpreted as N streams of
    /// scalars or pairs, where the streams may or may not be of equal length
    pub(crate) fn target_len(num_streams: usize, pairs: bool) -> impl Strategy<Value = usize> {
        // Decide how many streams will have more elements than the others
        debug_assert!(num_streams > 0);
        let num_longer_streams = if num_streams == 1 {
            Just(0).boxed()
        } else {
            prop_oneof![
                2 => Just(0),
                3 => 1..num_streams,
            ]
            .boxed()
        };

        // Decide the length of the shortest streams of scalars/pairs
        let scalars_per_stream_elem = 1 + pairs as usize;
        let num_substreams = num_streams * scalars_per_stream_elem;
        let default_sizes = SizeRange::default();
        let min_substream_len =
            (default_sizes.start() / num_substreams)..(default_sizes.end_excl() / num_substreams);

        // Deduce the overall dataset length
        (min_substream_len, num_longer_streams).prop_map(
            move |(min_substream_len, num_longer_streams)| {
                min_substream_len * num_substreams + num_longer_streams * scalars_per_stream_elem
            },
        )
    }

    /// Mostly-valid subnormal amounts, for a given total dataset length
    pub(crate) fn num_subnormals(target_len: usize) -> impl Strategy<Value = usize> {
        if target_len < 2 {
            prop_oneof![
                2 => 0..=target_len,
                3 => (target_len+1)..,
            ]
            .boxed()
        } else {
            prop_oneof![
                1 => Just(0),
                3 => 1..target_len,
                1 => Just(target_len),
                1 => (target_len+1)..,
            ]
            .boxed()
        }
    }

    /// Inputs for an input generator test
    pub fn target_and_num_subnormals(ilp: usize) -> impl Strategy<Value = (Vec<f32>, usize)> {
        target_len(ilp, false).prop_flat_map(|target_len| {
            (
                prop::collection::vec(any::<f32>(), target_len),
                num_subnormals(target_len),
            )
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{test_utils::*, *};
    use crate::tests::assert_panics;
    use proptest::{prelude::*, sample::SizeRange};
    use std::{
        cell::RefCell,
        ops::{Deref, Range},
        panic::AssertUnwindSafe,
        ptr::NonNull,
        rc::Rc,
    };

    /// Inputs for [`GeneratorStream`] implementation tests
    pub fn stream_target_subnormals() -> impl Strategy<Value = (usize, usize, Vec<f32>, Vec<bool>)>
    {
        num_streams_and_target(false).prop_flat_map(|(num_streams, target)| {
            let target_len = target.len();
            (
                0..num_streams,
                Just(num_streams),
                Just(target),
                prop::collection::vec(any::<bool>(), target_len.div_ceil(num_streams)),
            )
        })
    }

    /// Reasonable arguments to `subnormal_picker` and `generate_elements`
    fn subnormal_config() -> impl Strategy<Value = (usize, usize)> {
        (
            0..SizeRange::default().end_excl(),
            Range::from(SizeRange::default()),
        )
    }

    proptest! {
        /// Test the subnormal number picker
        #[test]
        fn subnormal_picker(
            (num_subnormals, target_len) in subnormal_config(),
        ) {
            // Creation should panic if num_subnormals > target_len
            let make_picker = || super::subnormal_picker(num_subnormals, target_len);
            if num_subnormals > target_len {
                return assert_panics(make_picker);
            }
            let mut picker = make_picker();

            // Picker should yield the expected number of subnormals
            let mut rng = rand::thread_rng();
            let mut actual_subnormals = 0;
            for _ in 0..target_len {
                actual_subnormals += picker(&mut rng) as usize;
            }
            prop_assert_eq!(actual_subnormals, num_subnormals);

            // In debug builds, excessive calls should be checked
            if cfg!(debug_assertions) {
                assert_panics(AssertUnwindSafe(|| picker(&mut rng)))?;
            }
        }
    }

    /// Given a reference to a slice element and the initial pointer of that
    /// slice (as given by `slice.as_ptr()`), tell which element it its
    fn index_of<T>(elem: &mut T, start: *const T) -> usize {
        debug_assert!(start.is_aligned());
        (elem as *mut T as usize - start as usize) / std::mem::size_of::<T>()
    }

    /// Expected indices of the elements of a stream in the matching subslice
    fn expected_stream_indices(
        stream_idx: usize,
        num_streams: usize,
    ) -> impl Iterator<Item = usize> {
        (0..).skip(stream_idx).step_by(num_streams)
    }

    /// Like [`target_len()`] but also generates the number of streams
    fn num_streams_and_target_len(pairs: bool) -> impl Strategy<Value = (usize, usize)> {
        (1usize..=MIN_FLOAT_REGISTERS)
            .prop_flat_map(move |num_streams| (Just(num_streams), target_len(num_streams, pairs)))
    }

    /// Dataset that can be evenly cut into N streams of scalars or pairs
    pub fn num_streams_and_target<T: Arbitrary>(
        pairs: bool,
    ) -> impl Strategy<Value = (usize, Vec<T>)> {
        num_streams_and_target_len(pairs).prop_flat_map(|(num_streams, target_len)| {
            (
                Just(num_streams),
                prop::collection::vec(any::<T>(), target_len),
            )
        })
    }

    proptest! {
        /// Test scalar iteration over data streams
        #[test]
        fn stream_scalar_iter((num_streams, mut target) in num_streams_and_target::<u8>(false)) {
            let initial = target.clone();
            let target = &mut target[..];
            let left_start = target.as_ptr();
            let min_elems = target.len() / num_streams;
            for stream_idx in 0..num_streams {
                let mut num_elems = 0;
                for (elem, expected_idx) in (DataStream {
                    target,
                    stream_idx,
                    num_streams,
                })
                .into_scalar_iter()
                .zip(expected_stream_indices(stream_idx, num_streams))
                {
                    prop_assert_eq!(index_of(elem, left_start), expected_idx);
                    num_elems += 1;
                }
                let has_extra_elem = stream_idx < (target.len() % num_streams);
                prop_assert_eq!(num_elems, min_elems + has_extra_elem as usize);
            }
            prop_assert_eq!(target, initial);
        }

        /// Test pairwise iteration over data streams
        #[test]
        fn stream_pair_iter((num_streams, mut target) in num_streams_and_target::<u8>(true)) {
            let initial = target.clone();
            let target = &mut target[..];
            debug_assert_eq!(target.len() % 2, 0);
            let half_len = target.len() / 2;
            let left_start = target.as_ptr();
            let right_start = left_start.wrapping_add(half_len);
            let min_elems = half_len / num_streams;
            for stream_idx in 0..num_streams {
                let mut num_elems = 0;
                for ([left, right], expected_idx) in (DataStream {
                    target,
                    stream_idx,
                    num_streams,
                })
                .into_pair_iter()
                .zip(expected_stream_indices(stream_idx, num_streams))
                {
                    prop_assert_eq!(index_of(left, left_start), expected_idx);
                    prop_assert_eq!(index_of(right, right_start), expected_idx);
                    num_elems += 1;
                }
                let has_extra_elem = stream_idx < (half_len % num_streams);
                prop_assert_eq!(num_elems, min_elems + has_extra_elem as usize);
            }
            prop_assert_eq!(target, initial);
        }
    }

    /// Test access to a particular stream element
    fn num_streams_target_and_idx() -> impl Strategy<Value = (usize, Vec<u8>, usize)> {
        num_streams_and_target(false).prop_flat_map(|(num_streams, target)| {
            let global_idx = if target.is_empty() {
                any::<usize>().boxed()
            } else {
                prop_oneof![
                    4 => 0..target.len(),
                    1 => target.len()..usize::MAX,
                ]
                .boxed()
            };
            (Just(num_streams), Just(target), global_idx)
        })
    }
    //
    proptest! {
        #[test]
        fn stream_scalar_at((num_streams, mut target, global_idx) in num_streams_target_and_idx()) {
            let initial = target.clone();
            let target = &mut target[..];
            let target_len = target.len();
            let start = target.as_ptr();
            let stream_idx = global_idx % num_streams;
            let mut stream = DataStream {
                target,
                stream_idx,
                num_streams,
            };
            let mut output_idx = || index_of(stream.scalar_at(global_idx), start);
            if global_idx >= target_len {
                return assert_panics(AssertUnwindSafe(output_idx));
            }
            prop_assert_eq!(output_idx(), global_idx);
            prop_assert_eq!(target, initial);
        }
    }

    /// Mock of [`GeneratorStream`] that records the sequence of method calls
    struct GeneratorStreamMock<T: FloatLike>(GeneratorStreamActionList<T>);
    //
    /// Recorded actions from a [`GeneratorStreamMock`]
    type GeneratorStreamActionList<T> = Rc<RefCell<Vec<GeneratorStreamAction<T>>>>;
    //
    /// Individual record from a [`GeneratorStreamActionList`]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum GeneratorStreamAction<T> {
        RecordSubnormal,
        GenerateNormal {
            global_idx: usize,
            narrow: T,
        },
        Finalize {
            target: NonNull<[T]>,
            stream_idx: usize,
            num_streams: usize,
        },
    }
    //
    impl<T: FloatLike> GeneratorStreamMock<T> {
        /// Set up the mock
        fn new() -> Self {
            Self(Rc::new(RefCell::new(Vec::new())))
        }

        /// Get temporary access to the stream's internal action list
        fn actions(
            &self,
        ) -> impl Deref<Target = impl Deref<Target = [GeneratorStreamAction<T>]>> + '_ {
            self.0.borrow()
        }

        /// Get access to the internal actions list that outlives `self`
        ///
        /// Useful for test scenarios where the stream is consumed by
        /// `finalize()` before we got a chance to investigate it.
        fn actions_rc(&self) -> GeneratorStreamActionList<T> {
            self.0.clone()
        }

        /// Get a copy of the stream's internal action list
        fn actions_clone(&self) -> Box<[GeneratorStreamAction<T>]> {
            self.0.borrow().iter().copied().collect()
        }
    }
    //
    impl<R: Rng, T: FloatLike> GeneratorStream<R> for GeneratorStreamMock<T> {
        type Float = T;

        fn record_subnormal(&mut self) {
            self.0
                .borrow_mut()
                .push(GeneratorStreamAction::RecordSubnormal);
        }

        fn generate_normal(
            &mut self,
            global_idx: usize,
            rng: &mut R,
            mut narrow: impl FnMut(&mut R) -> Self::Float,
        ) -> Self::Float {
            let narrow = narrow(rng);
            self.0
                .borrow_mut()
                .push(GeneratorStreamAction::GenerateNormal { global_idx, narrow });
            narrow
        }

        fn finalize(self, stream: DataStream<'_, Self::Float>, _rng: &mut R) {
            self.0.borrow_mut().push(GeneratorStreamAction::Finalize {
                target: NonNull::from(stream.target),
                stream_idx: stream.stream_idx,
                num_streams: stream.num_streams,
            });
        }
    }

    /// Inputs for an [`element_generator()`] test
    fn num_streams_and_subnormal_config(
        pairs: bool,
    ) -> impl Strategy<Value = (usize, usize, usize)> {
        num_streams_and_target_len(pairs).prop_flat_map(|(num_streams, target_len)| {
            (
                Just(num_streams),
                num_subnormals(target_len),
                Just(target_len),
            )
        })
    }

    proptest! {
        /// Test generation of streams of scalar elements
        #[test]
        fn scalar_generator(
            (num_streams, num_subnormals, target_len) in num_streams_and_subnormal_config(false),
        ) {
            // Set up a mock of the expected element_generator usage context
            let mut rng = rand::thread_rng();
            if num_subnormals > target_len {
                return assert_panics(AssertUnwindSafe(|| {
                    #[allow(unused_must_use)]
                    super::element_generator::<_, GeneratorStreamMock<f32>>(num_subnormals, target_len, &mut rng);
                }));
            }
            let mut generator = super::element_generator(num_subnormals, target_len, &mut rng);
            let mut streams =
                std::iter::repeat_with(GeneratorStreamMock::<f32>::new).take(num_streams).collect::<Box<[_]>>();

            // Simulate producing N interleaved streams of data
            let mut recorded_subnormals = 0;
            'outer: for first_idx in (0..target_len).step_by(num_streams) {
                for (offset, stream) in streams.iter_mut().enumerate() {
                    // Decide if this element should be generated
                    let elem_idx = first_idx + offset;
                    if elem_idx >= target_len {
                        break 'outer;
                    }

                    // Back up previous stream action list
                    let prev_actions = stream.actions_clone();

                    // Make sure the generator does only one thing to the stream
                    generator(stream, elem_idx);
                    let actions = stream.actions();
                    prop_assert_eq!(actions.len(), prev_actions.len() + 1);
                    let new_action = actions.last().unwrap();

                    // Make sure it does something expected
                    match *new_action {
                        GeneratorStreamAction::RecordSubnormal => recorded_subnormals += 1,
                        GeneratorStreamAction::GenerateNormal { global_idx, narrow } => {
                            prop_assert_eq!(global_idx, elem_idx);
                            prop_assert!(narrow >= 0.5);
                            prop_assert!(narrow < 2.0);
                        }
                        GeneratorStreamAction::Finalize { .. } => unreachable!(),
                    }
                }
            }

            // Check that the right number of subnormal numbers was produced
            prop_assert_eq!(recorded_subnormals, num_subnormals);
        }

        /// Test generation of streams of pairs
        #[test]
        fn pair_generator(
            (num_streams, num_subnormals, target_len) in num_streams_and_subnormal_config(true),
        ) {
            // Set up a mock of the expected element_generator usage context
            debug_assert_eq!(target_len % 2, 0);
            let half_len = target_len / 2;
            let mut rng = rand::thread_rng();
            if num_subnormals > target_len {
                return assert_panics(AssertUnwindSafe(|| {
                    #[allow(unused_must_use)]
                    super::element_generator::<_, GeneratorStreamMock<f32>>(num_subnormals, target_len, &mut rng);
                }));
            }
            let mut generator = super::element_generator(num_subnormals, target_len, &mut rng);
            let mut streams =
                std::iter::repeat_with(GeneratorStreamMock::<f32>::new).take(num_streams).collect::<Box<[_]>>();

            // Simulate producing N interleaved streams of data
            let mut recorded_subnormals = 0;
            'outer: for first_left_idx in (0..half_len).step_by(num_streams) {
                for (offset, stream) in streams.iter_mut().enumerate() {
                    // Compute the index of each pair element
                    let left_idx = first_left_idx + offset;
                    let right_idx = left_idx + half_len;

                    // Decide if this pair should be generated
                    if right_idx >= target_len {
                        break 'outer;
                    }
                    debug_assert!(left_idx < half_len);

                    // Generate each element of the pair
                    for elem_idx in [left_idx, right_idx] {
                        // Back up previous stream action list
                        let prev_actions = stream.actions_clone();

                        // Make sure the generator does only one thing to the stream
                        generator(stream, elem_idx);
                        let actions = stream.actions();
                        prop_assert_eq!(actions.len(), prev_actions.len() + 1);
                        let new_action = actions.last().unwrap();

                        // Make sure it does something expected
                        match *new_action {
                            GeneratorStreamAction::RecordSubnormal => recorded_subnormals += 1,
                            GeneratorStreamAction::GenerateNormal { global_idx, narrow } => {
                                prop_assert_eq!(global_idx, elem_idx);
                                prop_assert!(narrow >= 0.5);
                                prop_assert!(narrow < 2.0);
                            }
                            GeneratorStreamAction::Finalize { .. } => unreachable!(),
                        }
                    }
                }
            }

            // Check that the right number of subnormal numbers was produced
            prop_assert_eq!(recorded_subnormals, num_subnormals);
        }
    }

    /// Mock of [`InputGenerator`] that records all actions from all streams
    struct InputGeneratorMock<T: FloatLike>(InputGeneratorStreams<T>);
    //
    /// Recorded per-stream actions from an [`InputGeneratorMock`]
    type InputGeneratorStreams<T> = Rc<RefCell<Vec<GeneratorStreamActionList<T>>>>;
    //
    impl<T: FloatLike> InputGeneratorMock<T> {
        /// Set up a new input generator mock
        fn new() -> Self {
            Self(Rc::new(RefCell::new(Vec::new())))
        }

        /// Get access to the per-stream actions list that outlives `self`
        ///
        /// Useful for test scenarios where the input generator is consumed
        /// before we got a chance to investigate it.
        fn stream_actions_rc(&self) -> InputGeneratorStreams<T> {
            self.0.clone()
        }
    }
    //
    impl<T: FloatLike, R: Rng> InputGenerator<T, R> for InputGeneratorMock<T> {
        type Stream<'a> = GeneratorStreamMock<T>;

        fn new_stream(&self) -> Self::Stream<'_> {
            let stream = GeneratorStreamMock::new();
            self.0.borrow_mut().push(stream.actions_rc());
            stream
        }
    }

    /// Inputs for a `generate_input` test that operates over borrowed memory
    fn memory_input<const ILP: usize>(pairs: bool) -> impl Strategy<Value = (Vec<f32>, usize)> {
        target_len(ILP, pairs).prop_flat_map(|target_len| {
            (
                prop::collection::vec(any::<f32>(), target_len),
                num_subnormals(target_len),
            )
        })
    }

    /// Inputs for a `generate_input` test that operates over owned registers
    fn registers_input<const INPUT_REGISTERS: usize>(
        pairs: bool,
    ) -> impl Strategy<Value = ([f32; INPUT_REGISTERS], usize)> {
        if pairs {
            assert_eq!(INPUT_REGISTERS % 2, 0);
        }
        (
            [any::<f32>(); INPUT_REGISTERS],
            num_subnormals(INPUT_REGISTERS),
        )
    }

    /// Test `generate_input_streams` in a certain configuration
    fn test_generate_input_streams<Storage: InputsMut<Element = f32>, const ILP: usize>(
        mut target: Storage,
        num_subnormals: usize,
    ) -> Result<(), TestCaseError> {
        // Set up a mock input generator that tracks per-stream actions
        let generator = InputGeneratorMock::new();
        let stream_actions = generator.stream_actions_rc();

        // Run generate_input_streams
        let generate = |target: &mut Storage| {
            generate_input_streams::<_, _, ILP>(
                target,
                &mut rand::thread_rng(),
                num_subnormals,
                generator,
            )
        };
        if num_subnormals > target.as_ref().len() {
            return assert_panics(AssertUnwindSafe(|| generate(&mut target)));
        }
        generate(&mut target);

        // Validate the results, using polymorphized code to reduce code bloat
        let expected_streams = if Storage::KIND.is_reused() { 1 } else { ILP };
        fn validate(
            target: &[f32],
            num_subnormals: usize,
            stream_actions: InputGeneratorStreams<f32>,
            expected_streams: usize,
        ) -> Result<(), TestCaseError> {
            // Check that generate_input_streams created at least the expected
            // number of data streams
            let streams = stream_actions.borrow();
            prop_assert!(streams.len() >= expected_streams);

            // Check that the stream generators performed the expected actions
            let min_stream_len = target.len() / expected_streams;
            let mut actual_subnormals = 0;
            for (stream_idx, actions) in streams.iter().map(|stream| stream.borrow()).enumerate() {
                // Any stream created after the expected ones should be a
                // placeholder memory value that is never used for anything
                if stream_idx >= expected_streams {
                    prop_assert!(actions.is_empty());
                    continue;
                }

                // Other streams should have one action per generated data
                // element, plus a trailing finalization action at the end.
                let has_extra_elem = stream_idx < (target.len() % expected_streams);
                let expected_elems = min_stream_len + has_extra_elem as usize;
                prop_assert_eq!(actions.len(), expected_elems + 1);
                let (last_action, other_actions) = actions.split_last().unwrap();
                prop_assert_eq!(
                    *last_action,
                    GeneratorStreamAction::Finalize {
                        target: NonNull::from(target),
                        stream_idx,
                        num_streams: expected_streams
                    }
                );

                // Check that the data element generation actions look correct
                let mut expected_idx = stream_idx;
                for action in other_actions {
                    debug_assert!(expected_idx < target.len());
                    match *action {
                        GeneratorStreamAction::RecordSubnormal => {
                            actual_subnormals += 1;
                            prop_assert!(target[expected_idx].is_subnormal());
                        }
                        GeneratorStreamAction::GenerateNormal { narrow, global_idx } => {
                            prop_assert!(narrow >= 0.5);
                            prop_assert!(narrow < 2.0);
                            prop_assert_eq!(global_idx, expected_idx);
                            prop_assert_eq!(target[expected_idx], narrow);
                        }
                        GeneratorStreamAction::Finalize { .. } => unreachable!(),
                    }
                    expected_idx += expected_streams;
                }
            }
            prop_assert_eq!(actual_subnormals, num_subnormals);
            Ok(())
        }
        validate(
            target.as_ref(),
            num_subnormals,
            stream_actions,
            expected_streams,
        )
    }
    //
    proptest! {
        #[test]
        fn generate_input_streams_memory_ilp1(
            (mut target, num_subnormals) in memory_input::<1>(false)
        ) {
            test_generate_input_streams::<_, 1>(&mut target[..], num_subnormals)?;
        }

        #[test]
        fn generate_input_streams_memory_ilp2(
            (mut target, num_subnormals) in memory_input::<2>(false)
        ) {
            test_generate_input_streams::<_, 2>(&mut target[..], num_subnormals)?;
        }

        #[test]
        fn generate_input_streams_memory_ilp3(
            (mut target, num_subnormals) in memory_input::<3>(false)
        ) {
            test_generate_input_streams::<_, 3>(&mut target[..], num_subnormals)?;
        }

        #[test]
        fn generate_input_streams_1reg_ilp1(
            (target, num_subnormals) in registers_input::<1>(false)
        ) {
            test_generate_input_streams::<_, 1>(target, num_subnormals)?;
        }

        #[test]
        fn generate_input_streams_1reg_ilp2(
            (target, num_subnormals) in registers_input::<1>(false)
        ) {
            test_generate_input_streams::<_, 2>(target, num_subnormals)?;
        }

        #[test]
        fn generate_input_streams_1reg_ilp3(
            (target, num_subnormals) in registers_input::<1>(false)
        ) {
            test_generate_input_streams::<_, 3>(target, num_subnormals)?;
        }

        #[test]
        fn generate_input_streams_2regs_ilp1(
            (target, num_subnormals) in registers_input::<2>(false)
        ) {
            test_generate_input_streams::<_, 1>(target, num_subnormals)?;
        }

        #[test]
        fn generate_input_streams_2regs_ilp2(
            (target, num_subnormals) in registers_input::<2>(false)
        ) {
            test_generate_input_streams::<_, 2>(target, num_subnormals)?;
        }

        #[test]
        fn generate_input_streams_2regs_ilp3(
            (target, num_subnormals) in registers_input::<2>(false)
        ) {
            test_generate_input_streams::<_, 3>(target, num_subnormals)?;
        }

        #[test]
        fn generate_input_streams_3regs_ilp1(
            (target, num_subnormals) in registers_input::<3>(false)
        ) {
            test_generate_input_streams::<_, 1>(target, num_subnormals)?;
        }

        #[test]
        fn generate_input_streams_3regs_ilp2(
            (target, num_subnormals) in registers_input::<3>(false)
        ) {
            test_generate_input_streams::<_, 2>(target, num_subnormals)?;
        }

        #[test]
        fn generate_input_streams_3regs_ilp3(
            (target, num_subnormals) in registers_input::<3>(false)
        ) {
            test_generate_input_streams::<_, 3>(target, num_subnormals)?;
        }
    }

    /// Test `generate_input_pairs` in a certain configuration
    fn test_generate_input_pairs<Storage: InputsMut<Element = f32>, const ILP: usize>(
        mut target: Storage,
        num_subnormals: usize,
    ) -> Result<(), TestCaseError> {
        // Set up a mock input generator that tracks per-stream actions
        let generator = InputGeneratorMock::new();
        let stream_actions = generator.stream_actions_rc();

        // Run generate_input_pairs
        let generate = |target: &mut Storage| {
            generate_input_pairs::<_, _, ILP>(
                target,
                &mut rand::thread_rng(),
                num_subnormals,
                generator,
            )
        };
        if num_subnormals > target.as_ref().len() {
            return assert_panics(AssertUnwindSafe(|| generate(&mut target)));
        }
        generate(&mut target);

        // Validate the results, using polymorphized code to reduce code bloat
        let expected_streams = if Storage::KIND.is_reused() { 1 } else { ILP };
        fn validate(
            target: &[f32],
            num_subnormals: usize,
            stream_actions: InputGeneratorStreams<f32>,
            expected_streams: usize,
        ) -> Result<(), TestCaseError> {
            // Check that generate_input_streams created at least the expected
            // number of data streams
            let streams = stream_actions.borrow();
            prop_assert!(streams.len() >= expected_streams);

            // Check that the stream generators performed the expected actions
            debug_assert_eq!(target.len() % 2, 0);
            let half_len = target.len() / 2;
            let min_pairs_per_stream = half_len / expected_streams;
            let mut actual_subnormals = 0;
            for (stream_idx, actions) in streams.iter().map(|stream| stream.borrow()).enumerate() {
                // Any stream created after the expected ones should be a
                // placeholder memory value that is never used for anything
                if stream_idx >= expected_streams {
                    prop_assert!(actions.is_empty());
                    continue;
                }

                // Other streams should have one action per generated data
                // element, plus a trailing finalization action at the end.
                let has_extra_pair = stream_idx < (half_len % expected_streams);
                let expected_pairs = min_pairs_per_stream + has_extra_pair as usize;
                prop_assert_eq!(actions.len(), 2 * expected_pairs + 1);
                let (last_action, other_actions) = actions.split_last().unwrap();
                prop_assert_eq!(
                    *last_action,
                    GeneratorStreamAction::Finalize {
                        target: NonNull::from(target),
                        stream_idx,
                        num_streams: expected_streams
                    }
                );

                // Check that the data element generation actions look correct
                let mut expected_left_idx = stream_idx;
                debug_assert_eq!(other_actions.len() % 2, 0);
                for action_pair in other_actions.chunks_exact(2) {
                    let expected_indices = [expected_left_idx, expected_left_idx + half_len];
                    for (action, expected_idx) in action_pair.iter().zip(expected_indices) {
                        debug_assert!(expected_idx < target.len());
                        match *action {
                            GeneratorStreamAction::RecordSubnormal => {
                                actual_subnormals += 1;
                                prop_assert!(target[expected_idx].is_subnormal());
                            }
                            GeneratorStreamAction::GenerateNormal { narrow, global_idx } => {
                                prop_assert!(narrow >= 0.5);
                                prop_assert!(narrow < 2.0);
                                prop_assert_eq!(global_idx, expected_idx);
                                prop_assert_eq!(target[expected_idx], narrow);
                            }
                            GeneratorStreamAction::Finalize { .. } => unreachable!(),
                        }
                    }
                    expected_left_idx += expected_streams;
                }
            }
            prop_assert_eq!(actual_subnormals, num_subnormals);
            Ok(())
        }
        validate(
            target.as_ref(),
            num_subnormals,
            stream_actions,
            expected_streams,
        )
    }
    //
    proptest! {
        #[test]
        fn generate_input_pairs_memory_ilp1(
            (mut target, num_subnormals) in memory_input::<1>(true)
        ) {
            test_generate_input_pairs::<_, 1>(target.as_mut_slice(), num_subnormals)?;
        }

        #[test]
        fn generate_input_pairs_memory_ilp2(
            (mut target, num_subnormals) in memory_input::<2>(true)
        ) {
            test_generate_input_pairs::<_, 2>(target.as_mut_slice(), num_subnormals)?;
        }

        #[test]
        fn generate_input_pairs_memory_ilp3(
            (mut target, num_subnormals) in memory_input::<3>(true)
        ) {
            test_generate_input_pairs::<_, 3>(target.as_mut_slice(), num_subnormals)?;
        }

        #[test]
        fn generate_input_pairs_2regs_ilp1(
            (target, num_subnormals) in registers_input::<2>(true)
        ) {
            test_generate_input_pairs::<_, 1>(target, num_subnormals)?;
        }

        #[test]
        fn generate_input_pairs_2regs_ilp2(
            (target, num_subnormals) in registers_input::<2>(true)
        ) {
            test_generate_input_pairs::<_, 2>(target, num_subnormals)?;
        }

        #[test]
        fn generate_input_pairs_2regs_ilp3(
            (target, num_subnormals) in registers_input::<2>(true)
        ) {
            test_generate_input_pairs::<_, 3>(target, num_subnormals)?;
        }

        #[test]
        fn generate_input_pairs_4regs_ilp1(
            (target, num_subnormals) in registers_input::<4>(true)
        ) {
            test_generate_input_pairs::<_, 1>(target, num_subnormals)?;
        }

        #[test]
        fn generate_input_pairs_4regs_ilp2(
            (target, num_subnormals) in registers_input::<4>(true)
        ) {
            test_generate_input_pairs::<_, 2>(target, num_subnormals)?;
        }

        #[test]
        fn generate_input_pairs_4regs_ilp3(
            (target, num_subnormals) in registers_input::<4>(true)
        ) {
            test_generate_input_pairs::<_, 3>(target, num_subnormals)?;
        }

        #[test]
        fn generate_input_pairs_6regs_ilp1(
            (target, num_subnormals) in registers_input::<6>(true)
        ) {
            test_generate_input_pairs::<_, 1>(target, num_subnormals)?;
        }

        #[test]
        fn generate_input_pairs_6regs_ilp2(
            (target, num_subnormals) in registers_input::<6>(true)
        ) {
            test_generate_input_pairs::<_, 2>(target, num_subnormals)?;
        }

        #[test]
        fn generate_input_pairs_6regs_ilp3(
            (target, num_subnormals) in registers_input::<6>(true)
        ) {
            test_generate_input_pairs::<_, 3>(target, num_subnormals)?;
        }
    }
}
