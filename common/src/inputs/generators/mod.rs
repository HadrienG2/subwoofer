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
        // Set up the interleaved data streams
        let mut streams: [_; MIN_FLOAT_REGISTERS] = std::array::from_fn(|_| generator.new_stream());
        assert!(num_streams <= streams.len());
        let streams = &mut streams[..num_streams];

        // Generate each element of each data stream
        let mut generate_element = element_generator(target.len(), rng, num_subnormals);
        let mut target_chunks = target.chunks_exact_mut(num_streams);
        for chunk in target_chunks.by_ref() {
            for (target, stream) in chunk.iter_mut().zip(streams.iter_mut()) {
                *target = generate_element(stream);
            }
        }
        for (target, stream) in target_chunks
            .into_remainder()
            .iter_mut()
            .zip(streams.iter_mut())
        {
            *target = generate_element(stream);
        }

        // Tweak final data streams to enforce a perfect accumulator reset
        for (stream_idx, stream) in streams
            .iter_mut()
            .map(|stream| std::mem::replace(stream, generator.new_stream()))
            .enumerate()
        {
            stream.finalize(DataStream {
                target,
                stream_idx,
                num_streams,
            })
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
        let mut generate_element = element_generator(target_len, rng, num_subnormals);
        let mut left_chunks = left_target.chunks_exact_mut(num_streams);
        let mut right_chunks = right_target.chunks_exact_mut(num_streams);
        'chunks: loop {
            match (left_chunks.next(), right_chunks.next()) {
                (Some(left_chunk), Some(right_chunk)) => {
                    for ((left_elem, right_elem), stream) in left_chunk
                        .iter_mut()
                        .zip(right_chunk)
                        .zip(streams.iter_mut())
                    {
                        *left_elem = generate_element(stream);
                        *right_elem = generate_element(stream);
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
                        *left_elem = generate_element(stream);
                        *right_elem = generate_element(stream);
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

        // Tweak final data streams to enforce a perfect accumulator reset
        for (stream_idx, stream) in streams
            .iter_mut()
            .map(|stream| std::mem::replace(stream, generator.new_stream()))
            .enumerate()
        {
            stream.finalize(DataStream {
                target,
                stream_idx,
                num_streams,
            })
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
    fn finalize(self, stream: DataStream<'_, Self::Float>);
}

/// Single input data stream in the context of a global data set
///
/// Can be interpreted either as a stream of scalar elements of type T, or pairs
/// thereof. See the `TypedStream`
pub struct DataStream<'data, T: FloatLike> {
    target: &'data mut [T],
    stream_idx: usize,
    num_streams: usize,
}
//
impl<'data, T: FloatLike> DataStream<'data, T> {
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
/// Given the total number of elements to be produced, the number of subnormal
/// inputs within this global dataset, and a random number generator, this
/// produces a function that, given the state of a particular substream of the
/// dataset, produces the next element of this substream.
fn element_generator<R: Rng, S: GeneratorStream<R>>(
    target_len: usize,
    rng: &mut R,
    num_subnormals: usize,
) -> impl FnMut(&mut S) -> S::Float + '_ {
    assert!(num_subnormals <= target_len);
    let narrow = floats::narrow_sampler::<S::Float, R>();
    let subnormal = floats::subnormal_sampler::<S::Float, R>();
    let mut pick_subnormal = subnormal_picker(target_len, num_subnormals);
    let mut global_idx = 0;
    move |state: &mut S| {
        debug_assert!(global_idx < target_len);
        let result = if pick_subnormal(rng) {
            state.record_subnormal();
            subnormal(rng)
        } else {
            state.generate_normal(global_idx, rng, &narrow)
        };
        global_idx += 1;
        result
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
    assert!(num_subnormals <= target_len);
    move |rng| {
        debug_assert!(target_len > 0);
        let subnormal_pos = rng.gen_range(0..target_len);
        let is_subnormal = subnormal_pos < num_subnormals;
        target_len -= 1;
        num_subnormals -= (is_subnormal) as usize;
        is_subnormal
    }
}
