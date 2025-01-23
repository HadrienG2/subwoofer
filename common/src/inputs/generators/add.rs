//! Input generation for benchmarks with an ADD/SUB pattern

use super::{DataStream, GeneratorStream, InputGenerator};
use crate::{floats::FloatLike, inputs::InputsMut};
use rand::prelude::*;

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
    super::generate_input_streams::<_, _, ILP>(target, rng, num_subnormals, AddGenerator);
}

/// (Lack of) global state for this input generator
struct AddGenerator;
//
impl<T: FloatLike, R: Rng> InputGenerator<T, R> for AddGenerator {
    type Stream<'a> = AddStream<T>;

    fn new_stream(&self) -> Self::Stream<'_> {
        AddStream::Unconstrained
    }
}

/// Per-stream state for this input generator
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
#[derive(Clone, Debug, PartialEq)]
enum AddStream<T: FloatLike> {
    /// No constraint on the next normal value
    Unconstrained,

    /// Next normal value should negate the effect of the previous one
    Negate {
        /// Previous normal input value
        value: T,

        /// Global position of this normal value in the generated data
        global_idx: usize,
    },
}
//
impl<T: FloatLike, R: Rng> GeneratorStream<R> for AddStream<T> {
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
                *self = Self::Negate { global_idx, value };
                value
            }
            Self::Negate { value, .. } => {
                *self = Self::Unconstrained;
                -value
            }
        }
    }

    fn finalize(self, mut stream: DataStream<'_, T>) {
        if let Self::Negate { global_idx, .. } = self {
            *stream.scalar_at(global_idx) = T::splat(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{floats, inputs::generators::tests::num_streams_and_target};
    use proptest::prelude::*;

    /// Test [`AddStream`]
    fn stream_target_subnormals() -> impl Strategy<Value = (usize, usize, Vec<f32>, Vec<bool>)> {
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
    //
    proptest! {
        #[test]
        fn add_stream((stream_idx, num_streams, mut target, subnormals) in stream_target_subnormals()) {
            // Set up a mock environment
            let rng = &mut rand::thread_rng();
            let mut narrow = floats::narrow_sampler();
            let subnormal = floats::subnormal_sampler();

            // Simulate the creation of a data stream, checking behavior
            let mut stream = AddStream::Unconstrained;
            let stream_indices = (0..target.len()).skip(stream_idx).step_by(num_streams);
            for (elem_idx, is_subnormal) in stream_indices.clone().zip(subnormals) {
                let prev_stream = stream.clone();
                let new_value = if is_subnormal {
                    <AddStream<_> as GeneratorStream<ThreadRng>>::record_subnormal(&mut stream);
                    prop_assert_eq!(&stream, &prev_stream);
                    subnormal(rng)
                } else {
                    let output: f32 = stream.generate_normal(elem_idx, rng, &mut narrow);
                    match (prev_stream, &stream) {
                        (AddStream::Unconstrained, AddStream::Negate { global_idx, value }) => {
                            prop_assert_eq!(*global_idx, elem_idx);
                            prop_assert!(*value >= 0.5);
                            prop_assert!(*value < 2.0);
                            prop_assert_eq!(output, *value);
                        }
                        (AddStream::Negate { global_idx: _, value }, AddStream::Unconstrained) => {
                            prop_assert_eq!(output, -value);
                        }
                        (from, to) => {
                            return Err(TestCaseError::fail(
                                format!("Did not expect a stream state transition from {from:?} to {to:?}")
                            ));
                        }
                    }
                    output
                };
                target[elem_idx] = new_value;
            }

            // Determine the expected final output
            let mut expected = target.clone();
            match &stream {
                AddStream::Unconstrained => {}
                AddStream::Negate { global_idx, value: _ } => expected[*global_idx] = 0.0,
            };

            // Check that this is actually the output we get
            <AddStream<_> as GeneratorStream<ThreadRng>>::finalize(
                stream,
                DataStream { target: &mut target, stream_idx, num_streams }
            );
            prop_assert!(
                target.into_iter().map(f32::to_bits)
                .eq(expected.into_iter().map(f32::to_bits))
            );
        }
    }

    // TODO: Test [`generate_add_inputs()`]
}
