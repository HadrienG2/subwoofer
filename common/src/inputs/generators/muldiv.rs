//! Input generator for benchmarks with a MUL/DIV pattern

use super::{DataStream, GeneratorStream, InputGenerator};
use crate::{
    floats::{self, suggested_extremal_bias, FloatLike},
    inputs::InputsMut,
};
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
    num_subnormals: usize,
    target: &mut Storage,
    rng: &mut impl Rng,
    inside_test: bool,
    invert_normal: impl Fn(Storage::Element) -> Storage::Element + 'static,
    cancel_subnormal: impl Fn(Storage::Element) -> Storage::Element + 'static,
) {
    super::generate_input_streams::<_, _, ILP>(
        num_subnormals,
        target,
        rng,
        inside_test,
        MulDivGenerator {
            invert_normal,
            cancel_subnormal,
            inside_test,
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

    /// Truth that we are generating data for a unit test
    inside_test: bool,

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

    fn finalize(self, stream: DataStream<'_, T>, rng: &mut R) {
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
                // magnitude, so it can only be tolerated as the last input of a
                // data stream if the resulting abnormal accumulator state is
                // (mostly) ignored by the next benchmark run because the first
                // input of the data stream is also subnormal.
                let first = stream
                    .next()
                    .expect("CancelSubnormal is only reachable after >= 1 subnormal inputs");
                if first.is_subnormal() {
                    return;
                }
                assert!(first.is_normal());

                // Otherwise we must isolate our trailing subnormal input...
                let last = stream.next_back().expect(
                    "We know the last input is subnormal, it cannot also be the first input if it's normal",
                );
                assert!(last.is_subnormal());

                // ...find the previous normal input in the data stream, coming
                // before all the other subnormal inputs that precede our
                // trailing subnormal input...
                let prev_normal = stream
                    .rev()
                    .chain(std::iter::once(first))
                    .find(|elem| elem.is_normal())
                    .expect("first is normal, so this predicate will be true at least once");

                // ...and shift our subnormal input there. It is safe to
                // overwrite a normal input that precedes a sequence of
                // subnormal inputs because subnormals destroy all previous
                // accumulator magnitude information.
                *prev_normal = *last;

                // The last input of the data stream will then be turned into a
                // normal input that recovers from the subnormals-induced
                // accumulator magnitude change. As a result, the accumulator
                // will be back to a narrow state after going through the full
                // data stream, as it should.
                let narrow =
                    floats::narrow_sampler(suggested_extremal_bias(self.generator.inside_test, 1));
                *last = (self.generator.cancel_subnormal)(narrow(rng));
            }
        }
    }
}

/// Test [`generate_muldiv_inputs()`] in a particular configuration
///
/// This is a common implementation detail of the unit tests of the `div_xyz`
/// and `mul_max` benchmarks, that should not be relied on by non-test code.
///
/// Compared to [`generate_muldiv_inputs`], it makes the additional assumption
/// that cancel_subnormal is a monotonic function, i.e. if we denote A and B to
/// inputs, then for all inputs `A <= x <= B` we expect `min(f(A), f(B)) <= f(x)
/// <= max(f(A), f(B))`.
#[cfg(any(test, feature = "unstable_test"))]
pub fn test_generate_muldiv_inputs<const ILP: usize>(
    num_subnormals: usize,
    target: &mut [f32],
    invert_normal: impl Fn(f32) -> f32 + Clone + 'static,
    cancel_subnormal: impl Fn(f32) -> f32 + Clone + 'static,
    integrate: impl Fn(f32, f32) -> f32 + 'static,
) -> Result<(), proptest::test_runner::TestCaseError> {
    // Imports specific to this test functionality
    use crate::{operations, test_utils::assert_panics};
    use proptest::prelude::*;
    use std::panic::AssertUnwindSafe;

    // Tolerance of the final check that the accumulator stayed in narrow range
    const NARROW_ACC_TOLERANCE: f32 = 5.0 * f32::EPSILON;
    let acc_range = (0.5 - NARROW_ACC_TOLERANCE)..=(2.0 + NARROW_ACC_TOLERANCE);

    // Generate the inputs
    let mut rng = rand::thread_rng();
    let generate = |mut target: &mut [f32], rng: &mut _| {
        generate_muldiv_inputs::<_, ILP>(
            num_subnormals,
            &mut target,
            rng,
            true,
            invert_normal.clone(),
            cancel_subnormal.clone(),
        )
    };
    if num_subnormals > target.len() {
        return assert_panics(AssertUnwindSafe(|| {
            generate(target, &mut rng);
        }));
    }
    generate(target, &mut rng);

    // Set up narrow accumulators
    let accs_init = operations::narrow_accumulators::<f32, ILP>(&mut rng, true);
    let mut accs = accs_init;

    // Check the result and simulate its effect on narrow accumulators
    let mut actual_subnormals = 0;
    let mut expected_state: [MulDivState<f32>; ILP] =
        std::array::from_fn(|_| MulDivState::Unconstrained);
    let mut should_be_end = [false; ILP];
    let error_context = |chunk_idx: usize, acc_idx| {
        format!(
            "\n\
            * Starting from initial accumulator {:?}\n\
            * After integrating input(s) {:?}\n",
            accs_init[acc_idx],
            target
                .iter()
                .skip(acc_idx)
                .step_by(ILP)
                .take(chunk_idx)
                .collect::<Vec<_>>()
        )
    };
    for (chunk_idx, chunk) in target.chunks(ILP).enumerate() {
        for (((acc_idx, &elem), acc), (expected_state, should_be_end)) in (chunk.iter().enumerate())
            .zip(&mut accs)
            .zip(expected_state.iter_mut().zip(&mut should_be_end))
        {
            let acc_before = *acc;
            let acc_after = integrate(*acc, elem);
            let error_context = || {
                format!(
                    "{}* While integrating next input {elem} into accumulator {acc_before}, with result {acc_after}\n",
                    error_context(chunk_idx, acc_idx)
                )
            };
            if elem.is_subnormal() {
                actual_subnormals += 1;
                *expected_state = MulDivState::CancelSubnormal;
            } else if elem == 1.0 {
                prop_assert_eq!(
                    expected_state,
                    &MulDivState::Unconstrained,
                    "{}",
                    error_context()
                );
                *should_be_end = true;
            } else {
                prop_assert!(elem.is_normal());
                prop_assert!(!*should_be_end);
                let expect_narrow_acc = || {
                    prop_assert!(
                        acc_range.contains(&acc_after),
                        "{}* New accumulator value escaped narrow range [0.5; 2] by more than tolerance {NARROW_ACC_TOLERANCE}",
                        error_context()
                    );
                    Ok(())
                };
                *expected_state = match *expected_state {
                    MulDivState::Unconstrained => {
                        prop_assert!(
                            (0.5..=2.0).contains(&elem),
                            "{}* Expected an input in narrow range [0.5; 2] but got {elem}",
                            error_context()
                        );
                        MulDivState::InvertNormal(elem)
                    }
                    MulDivState::InvertNormal(inverted) => {
                        let inverse = invert_normal(inverted);
                        prop_assert_eq!(
                            elem,
                            inverse,
                            "{}* Expected input {} that's the inverse of the previous input {}, but got {}",
                            error_context(),
                            inverse,
                            inverted,
                            elem
                        );
                        expect_narrow_acc()?;
                        MulDivState::Unconstrained
                    }
                    MulDivState::CancelSubnormal => {
                        let bound1 = cancel_subnormal(0.5);
                        let bound2 = cancel_subnormal(2.0);
                        let min = bound1.min(bound2);
                        let max = bound2.max(bound2);
                        prop_assert!(
                            (min..=max).contains(&elem),
                            "{}* Expected an input in subnormal recovery range [{min}; {max}] but got {elem}", error_context()
                        );
                        expect_narrow_acc()?;
                        MulDivState::Unconstrained
                    }
                };
            }
            *acc = acc_after;
        }
    }

    // Check that the input meets its goals of subnormal count and acc magnitude preservation
    prop_assert_eq!(actual_subnormals, num_subnormals);
    let first_inputs: [Option<f32>; ILP] = std::array::from_fn(|idx| target.get(idx).copied());
    for ((acc_idx, final_acc), first_input) in accs.into_iter().enumerate().zip(first_inputs) {
        if !first_input
            .unwrap_or(f32::MIN_POSITIVE / 2.0)
            .is_subnormal()
        {
            prop_assert!(
                acc_range.contains(&final_acc),
                "{}* Final accumulator value {final_acc} escaped narrow range [0.5; 2] by more than tolerance {NARROW_ACC_TOLERANCE}",
                error_context(target.len().div_ceil(ILP), acc_idx)
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        floats::{self, test_utils::suggested_extremal_bias},
        inputs::generators::test_utils::stream_target_subnormals,
    };
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
        fn new(expected_call_count: usize) -> (Self, impl Fn(f32) -> f32 + 'static) {
            let invocations = Rc::new(RefCell::new(Vec::new()));
            let callback = {
                let invocations = invocations.clone();
                let normal = floats::normal_sampler(suggested_extremal_bias(expected_call_count));
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
        fn mul_div_stream((stream_idx, num_streams, mut target, subnormals) in stream_target_subnormals(false)) {
            // Determine how many normal/subnormal numbers we expect to generate
            // and how many times we expect each callback to be called
            let mut expected_normal_outputs = 0;
            let mut expected_subnormal_outputs = 0;
            let mut expected_invert_normal_calls = 0;
            let mut expected_cancel_subnormal_calls = 0;
            {
                let mut invert_normal = false;
                let mut cancel_subnormal = false;
                for &is_subnormal in &subnormals {
                    if is_subnormal {
                        expected_subnormal_outputs += 1;
                        invert_normal = false;
                        cancel_subnormal = true;
                    } else {
                        expected_normal_outputs += 1;
                        if invert_normal {
                            expected_invert_normal_calls += 1;
                        } else if cancel_subnormal {
                            expected_cancel_subnormal_calls += 1;
                            cancel_subnormal = false;
                        }
                        invert_normal = !invert_normal;
                    }
                }
            }

            // Set up a mock input generator
            let (mut invert_normal_mock, invert_normal) = CallbackMock::new(expected_invert_normal_calls);
            let (mut cancel_subnormal_mock, cancel_subnormal) = CallbackMock::new(expected_cancel_subnormal_calls);
            let generator = MulDivGenerator {
                invert_normal,
                cancel_subnormal,
                inside_test: true,
                _unused: PhantomData,
            };

            // Set up a mock data stream
            let rng = &mut rand::thread_rng();
            let mut narrow = floats::narrow_sampler(suggested_extremal_bias(expected_normal_outputs));
            let subnormal = floats::subnormal_sampler(suggested_extremal_bias(expected_subnormal_outputs));
            let mut stream = <MulDivGenerator<f32, _, _> as InputGenerator<f32, ThreadRng>>::new_stream(&generator);
            prop_assert!(std::ptr::eq(stream.generator, &generator));
            prop_assert_eq!(&stream.state, &MulDivState::Unconstrained);

            // Simulate the generation of dataset, checking stream behavior
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
            let mut last_input_cancels_subnormal = false;
            'fixup: {
                let mut elements =
                    DataStream { target: &mut expected, stream_idx, num_streams }.into_scalar_iter();
                match &stream.state {
                    MulDivState::Unconstrained => {}
                    MulDivState::InvertNormal(_) => {
                        *elements.next_back().unwrap() = 1.0;
                    }
                    MulDivState::CancelSubnormal => {
                        // If the first item is subnormal, we don't care
                        let first = elements.next().unwrap();
                        if first.is_subnormal() {
                            break 'fixup;
                        }

                        // Otherwise, shift the subnormal input earlier in the
                        // data stream, before its subnormal neighbors...
                        let last = elements.next_back().unwrap();
                        let prev_normal = elements
                            .rev()
                            .chain(std::iter::once(first))
                            .find(|elem| elem.is_normal())
                            .expect("first is normal, so this predicate will be true at least once");
                        *prev_normal = *last;

                        // ...then remember that the last input must
                        // cancel a subnormal number
                        last_input_cancels_subnormal = true;
                    }
                }
            }

            // Compute the final output after a state backup
            <MulDivStream<_, _, _> as GeneratorStream<ThreadRng>>::finalize(
                stream,
                DataStream { target: &mut target, stream_idx, num_streams },
                rng,
            );

            // Check that we get the expected final state
            prop_assert!(invert_normal_mock.new_invocations().is_empty());
            let cancel_subnormal_invocations = cancel_subnormal_mock.new_invocations();
            if last_input_cancels_subnormal {
                prop_assert_eq!(cancel_subnormal_invocations.len(), 1);
                prop_assert!(cancel_subnormal_invocations[0].argument >= 0.5);
                prop_assert!(cancel_subnormal_invocations[0].argument < 2.0);
                let last_idx = stream_indices.last().unwrap();
                expected[last_idx] = cancel_subnormal_invocations[0].result;
            } else {
                prop_assert!(cancel_subnormal_invocations.is_empty());
            }
            prop_assert!(
                target.into_iter().map(f32::to_bits)
                .eq(expected.into_iter().map(f32::to_bits))
            );
        }
    }

    // High-level tests of `generate_muldiv_inputs` are not found here because
    // they depend on the choice of `invert_normal` and `cancel_subnormal`
    // callbacks, which is benchmark-specific. Instead, a generic test procedure
    // is exposed above, and used by the tests of each individual benchmark that
    // uses the "muldiv" input generation logic.
}
