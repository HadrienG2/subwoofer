//! Input generator for benchmarks with a MUL/DIV pattern

use super::{DataStream, GeneratorStream, InputGenerator};
use crate::{floats::FloatLike, inputs::InputsMut};
use rand::prelude::*;
use std::marker::PhantomData;

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
///     iteration of the next run and it doesn't matter much that it has an
///     unusual magnitude). This is achieved by enforcing that when the first
///     input value is normal and the last input value is subnormal, they are
///     swapped and what is now the new last normal input value is regenerated.
pub fn generate_muldiv_inputs<Storage: InputsMut, const ILP: usize>(
    target: &mut Storage,
    rng: &mut impl Rng,
    num_subnormals: usize,
    invert_normal: impl Fn(Storage::Element) -> Storage::Element + 'static,
    cancel_subnormal: impl Fn(Storage::Element) -> Storage::Element + 'static,
) {
    super::generate_input_streams::<_, _, ILP>(
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

/// Global state for this input generator
struct MulDivGenerator<T: FloatLike, InvertNormal, CancelSubnormal>
where
    InvertNormal: Fn(T) -> T,
    CancelSubnormal: Fn(T) -> T,
{
    /// Compute the inverse of a normal input
    invert_normal: InvertNormal,

    /// Modify a newly generated narrow input to cancel the effect of a previous
    /// subnormal input
    cancel_subnormal: CancelSubnormal,

    /// Mark T as used so rustc doesn't go mad
    _unused: PhantomData<T>,
}
//
impl<T: FloatLike, R: Rng, InvertNormal, CancelSubnormal> InputGenerator<T, R>
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
    /// Back-reference to the common generator state
    generator: &'generator MulDivGenerator<T, InvertNormal, CancelSubnormal>,

    /// Per-stream state machine
    state: MulDivState<T>,
}
//
/// Inner state machine of [`MulDivStream`]
#[derive(Clone, Debug, PartialEq)]
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
impl<T: FloatLike, R: Rng, InvertNormal, CancelSubnormal> GeneratorStream<R>
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

    fn finalize(self, stream: DataStream<'_, T>) {
        let mut stream = stream.into_scalar_iter();
        match self.state {
            MulDivState::Unconstrained => {}
            MulDivState::InvertNormal(_) => {
                let last = stream
                    .last()
                    .expect("InvertNormal is only reachable after >= 1 normal inputs");
                assert!(last.is_normal());
                *last = T::splat(1.0);
            }
            MulDivState::CancelSubnormal => {
                // A subnormal input brings the accumulator to an abnormal
                // magnitude, so it can only be tolerated as the last input
                // of the data stream if the resulting abnormal accumulator
                // state is ignored by the next benchmark run (because that
                // run starts with a subnormal input).
                let first = stream
                    .next()
                    .expect("CancelSubnormal is only reachable after >= 1 subnormal inputs");
                if first.is_subnormal() {
                    return;
                }
                assert!(first.is_normal());

                // Otherwise, we should exchange the last subnormal input
                // with the first normal input...
                let last = stream.next_back().expect(
                    "Last input should be subnormal, it cannot be the first input if it's normal",
                );
                assert!(last.is_subnormal());
                std::mem::swap(first, last);

                // ...and fix up that new trailing normal input so that it
                // is correct in the context of the previous normal input
                // values at the end of the stream.
                let mut previous_normals = stream.rev().take_while(|elem| elem.is_normal());
                if let Some(before_last) = previous_normals.next().copied() {
                    let num_before_that = previous_normals.count();
                    if num_before_that % 2 == 0 {
                        // If there is an even number of inputs before
                        // the second-to-last normal input, then this
                        // normal input grows/shrinks the accumulator
                        // and we should invert its effect to restore
                        // the accumulator back to its initial value.
                        *last = (self.generator.invert_normal)(before_last);
                    } else {
                        // If there is an odd number of inputs before
                        // the second-to-last normal input, then this
                        // second-to-last input brought the accumulator
                        // back to its initial [1/2; 2[ range, and the
                        // last normal input should keep the accumulator
                        // in the same order of magnitude.
                        *last = T::splat(1.0);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{floats, inputs::generators::tests::stream_target_subnormals};
    use proptest::prelude::*;
    use std::{cell::RefCell, rc::Rc};

    /// Mock of a callback that logs invocations
    struct CallbackMock(Rc<RefCell<Vec<CallbackInvocation>>>);
    //
    /// Log of a single call to a callback
    struct CallbackInvocation {
        argument: f32,
        result: f32,
    }
    //
    impl CallbackMock {
        /// Set up a callback mock
        fn new() -> (Self, impl Fn(f32) -> f32 + 'static) {
            let invocations = Rc::new(RefCell::new(Vec::new()));
            let callback = {
                let invocations = invocations.clone();
                let normal = floats::normal_sampler();
                move |argument| {
                    let mut invocations = invocations.borrow_mut();
                    let result = normal(&mut rand::thread_rng());
                    invocations.push(CallbackInvocation { argument, result });
                    result
                }
            };
            (Self(invocations), callback)
        }

        /// Fetch the list of callback invocations since the last call
        fn new_invocations(&mut self) -> Box<[CallbackInvocation]> {
            let mut invocations = self.0.borrow_mut();
            std::mem::take(&mut *invocations).into_boxed_slice()
        }
    }

    proptest! {
        /// Test [`MulDivStream`]
        #[test]
        fn mul_div_stream((stream_idx, num_streams, mut target, subnormals) in stream_target_subnormals()) {
            // Set up a mock input generator
            let (mut invert_normal_mock, invert_normal) = CallbackMock::new();
            let (mut cancel_subnormal_mock, cancel_subnormal) = CallbackMock::new();
            let generator = MulDivGenerator {
                invert_normal,
                cancel_subnormal,
                _unused: PhantomData,
            };

            // Set up a mock data stream
            let rng = &mut rand::thread_rng();
            let mut narrow = floats::narrow_sampler();
            let subnormal = floats::subnormal_sampler();
            let mut stream = <MulDivGenerator<f32, _, _> as InputGenerator<f32, ThreadRng>>::new_stream(&generator);

            // Simulate the generation of data stream, checking stream behavior
            let stream_indices = (0..target.len()).skip(stream_idx).step_by(num_streams);
            for (elem_idx, is_subnormal) in stream_indices.clone().zip(subnormals) {
                let new_value = if is_subnormal {
                    <MulDivStream<_, _, _> as GeneratorStream<ThreadRng>>::record_subnormal(&mut stream);
                    prop_assert_eq!(&stream.state, &MulDivState::CancelSubnormal);
                    subnormal(rng)
                } else {
                    let prev_state = stream.state.clone();
                    let output: f32 = stream.generate_normal(elem_idx, rng, &mut narrow);
                    match (prev_state, &stream.state) {
                        (MulDivState::Unconstrained, MulDivState::InvertNormal(value)) => {
                            prop_assert!(*value >= 0.5);
                            prop_assert!(*value < 2.0);
                            prop_assert_eq!(output, *value);
                        }
                        (MulDivState::InvertNormal(value), MulDivState::Unconstrained) => {
                            let invocations = invert_normal_mock.new_invocations();
                            prop_assert_eq!(invocations.len(), 1);
                            prop_assert_eq!(invocations[0].argument, value);
                            prop_assert_eq!(invocations[0].result, output);
                        }
                        (MulDivState::CancelSubnormal, MulDivState::Unconstrained) => {
                            let invocations = cancel_subnormal_mock.new_invocations();
                            prop_assert_eq!(invocations.len(), 1);
                            prop_assert!(invocations[0].argument >= 0.5);
                            prop_assert!(invocations[0].argument < 2.0);
                            prop_assert_eq!(invocations[0].result, output);
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
            prop_assert!(invert_normal_mock.new_invocations().is_empty());
            prop_assert!(cancel_subnormal_mock.new_invocations().is_empty());

            // Determine the expected final output
            let mut expected = target.clone();
            let last_input_inverts = 'fixup: {
                let mut elements =
                    DataStream { target: &mut expected, stream_idx, num_streams }.into_scalar_iter();
                match &stream.state {
                    MulDivState::Unconstrained => {
                        None
                    }
                    MulDivState::InvertNormal(_) => {
                        *elements.next_back().unwrap() = 1.0;
                        None
                    }
                    MulDivState::CancelSubnormal => {
                        // If the first item is subnormal, we don't care
                        let first = elements.next().unwrap();
                        if first.is_subnormal() {
                            break 'fixup None;
                        }

                        // Otherwise, put the subnormal input at the front...
                        let last = elements.next_back().unwrap();
                        std::mem::swap(first, last);

                        // ...then fixup new last normal number if needed to
                        // cancel out the effect of previous normal numbers
                        let mut previous_normals =
                            elements.rev().take_while(|elem| elem.is_normal());
                        if let Some(before_last) = previous_normals.next().copied() {
                            let num_previous = previous_normals.count();
                            if num_previous % 2 == 0 {
                                // Must cancel effect of before_last
                                Some(before_last)
                            } else {
                                // Must not introduce a new magnitude change
                                *last = 1.0;
                                None
                            }
                        } else {
                            None
                        }
                    }
                }
            };

            // Compute the final output after a state backup
            <MulDivStream<_, _, _> as GeneratorStream<ThreadRng>>::finalize(
                stream,
                DataStream { target: &mut target, stream_idx, num_streams }
            );

            // Check that we get the expected final state
            let invert_normal_invocations = invert_normal_mock.new_invocations();
            if let Some(last_input_inverts) = last_input_inverts {
                prop_assert_eq!(invert_normal_invocations.len(), 1);
                prop_assert_eq!(invert_normal_invocations[0].argument, last_input_inverts);
                let last_idx = stream_indices.last().unwrap();
                expected[last_idx] = invert_normal_invocations[0].result;
            } else {
                prop_assert!(invert_normal_invocations.is_empty());
            }
            prop_assert!(cancel_subnormal_mock.new_invocations().is_empty());
            prop_assert!(
                target.into_iter().map(f32::to_bits)
                .eq(expected.into_iter().map(f32::to_bits))
            );
        }
    }
}
