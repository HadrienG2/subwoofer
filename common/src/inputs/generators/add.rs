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

    fn finalize(self, mut stream: DataStream<'_, T>, _rng: &mut R) {
        if let Self::Negate { global_idx, .. } = self {
            *stream.scalar_at(global_idx) = T::splat(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        floats,
        inputs::generators::{
            test_utils::target_and_num_subnormals, tests::stream_target_subnormals,
        },
        tests::assert_panics,
    };
    use core::f32;
    use proptest::prelude::*;
    use std::panic::AssertUnwindSafe;

    proptest! {
        /// Test [`AddStream`]
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
                DataStream { target: &mut target, stream_idx, num_streams },
                rng,
            );
            prop_assert!(
                target.into_iter().map(f32::to_bits)
                .eq(expected.into_iter().map(f32::to_bits))
            );
        }
    }

    /// Test [`generate_add_inputs()`]
    fn test_generate_add_inputs<const ILP: usize>(
        target: &mut [f32],
        num_subnormals: usize,
    ) -> Result<(), TestCaseError> {
        // Generate the inputs
        let mut rng = rand::thread_rng();
        let generate = |mut target: &mut [f32], rng: &mut _| {
            generate_add_inputs::<_, ILP>(&mut target, rng, num_subnormals)
        };
        if num_subnormals > target.len() {
            return assert_panics(AssertUnwindSafe(|| {
                generate(target, &mut rng);
            }));
        }
        generate(target, &mut rng);

        // Set up narrow accumulators
        let narrow = floats::narrow_sampler();
        let accs_init: [f32; ILP] = std::array::from_fn(|_| narrow(&mut rng));

        // Check the result and simulate its effect on narrow accumulators
        let mut actual_subnormals = 0;
        let mut accs = accs_init;
        let mut to_negate: [Option<f32>; ILP] = [None; ILP];
        let mut last_normal = [false; ILP];
        for chunk in target.chunks(ILP) {
            for ((&elem, acc), (to_negate, last_normal)) in (chunk.iter())
                .zip(&mut accs)
                .zip(to_negate.iter_mut().zip(&mut last_normal))
            {
                if elem.is_subnormal() {
                    actual_subnormals += 1;
                } else if elem == 0.0 {
                    *last_normal = true;
                } else {
                    prop_assert!(elem.is_normal());
                    prop_assert!(!*last_normal);
                    if let Some(negated) = to_negate {
                        prop_assert_eq!(elem, -*negated);
                        *to_negate = None;
                    } else {
                        *to_negate = Some(elem)
                    }
                }
                *acc += elem;
            }
        }

        // Check that the input meets its goals of subnormal count and acc preservation
        prop_assert_eq!(actual_subnormals, num_subnormals);
        for (acc, init) in accs.iter().zip(accs_init) {
            let difference = (acc - init).abs();
            let threshold = 30.0 * f32::EPSILON * init;
            prop_assert!(
                difference < threshold,
                "{acc} is too far from {init} (difference {difference} above threshold {threshold} after integrating inputs {target:?})"
            )
        }
        Ok(())
    }
    //
    proptest! {
        #[test]
        fn generate_add_inputs_ilp1((mut target, num_subnormals) in target_and_num_subnormals(1)) {
            test_generate_add_inputs::<1>(&mut target, num_subnormals)?;
        }

        #[test]
        fn generate_add_inputs_ilp2((mut target, num_subnormals) in target_and_num_subnormals(2)) {
            test_generate_add_inputs::<2>(&mut target, num_subnormals)?;
        }

        #[test]
        fn generate_add_inputs_ilp3((mut target, num_subnormals) in target_and_num_subnormals(3)) {
            test_generate_add_inputs::<3>(&mut target, num_subnormals)?;
        }
    }
}
