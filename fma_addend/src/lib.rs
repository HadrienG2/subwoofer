use std::ops::Not;

use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{
        generators::{generate_input_pairs, DataStream, GeneratorStream, InputGenerator},
        Inputs, InputsMut,
    },
    operations::{self, Benchmark, BenchmarkRun, Operation},
};
use rand::Rng;

/// FMA with possibly subnormal addend
#[derive(Clone, Copy)]
pub struct FmaAddend;
//
impl Operation for FmaAddend {
    const NAME: &str = "fma_addend";

    // One register for the multiplier, one for its inverse
    fn aux_registers_regop(_input_registers: usize) -> usize {
        2
    }

    // acc is not reused after FMA, so memory operands can be used. Without
    // memory operands, need to load elem into another register before FMA.
    const AUX_REGISTERS_MEMOP: usize = 2 + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<Storage: InputsMut, const ILP: usize>(
        input_storage: Storage,
    ) -> impl Benchmark<Float = Storage::Element> {
        assert_eq!(
            input_storage.as_ref().len() % 2,
            0,
            "Invalid test input length"
        );
        FmaAddendBenchmark::<_, ILP> {
            input_storage,
            num_subnormals: None,
        }
    }
}

/// [`Benchmark`] of [`FmaAddend`]
struct FmaAddendBenchmark<Storage: InputsMut, const ILP: usize> {
    input_storage: Storage,
    num_subnormals: Option<usize>,
}
//
impl<Storage: InputsMut, const ILP: usize> Benchmark for FmaAddendBenchmark<Storage, ILP> {
    type Float = Storage::Element;

    fn num_operations(&self) -> usize {
        operations::accumulated_len(&self.input_storage, ILP)
    }

    fn setup_inputs(&mut self, num_subnormals: usize) {
        assert!(num_subnormals <= self.input_storage.as_ref().len());
        self.num_subnormals = Some(num_subnormals);
    }

    #[inline]
    fn start_run(&mut self, rng: &mut impl Rng, inside_test: bool) -> Self::Run<'_> {
        // This benchmark repeatedly multiplies the accumulator by a certain
        // multiplier and its inverse in order to exercise the "multiplier" part
        // of the FMA with normal values without getting an unbounded increase
        // or decrease of accumulator magnitude across benchmark iterations.
        //
        // Unfortunately, for this to work, the inverse must be exact, and this
        // only happens when the multiplier is a power of two. We'll avoid 1 as
        // a multiplier since the hardware is too likely to over-optimize for it
        // (because fma(1, x, y) trivially simplifies into x + y), so if we want
        // to only use numbers in the usual "narrow" [1/2; 2] range, this means
        // that our only remaining choices are 1/2 and 2.
        let multiplier = if rng.gen::<bool>() { 0.5 } else { 2.0 };
        let multiplier = pessimize::hide(Storage::Element::splat(multiplier));
        let inv_multiplier = pessimize::hide(Storage::Element::splat(1.0) / multiplier);

        // Generate input data
        generate_inputs::<_, ILP>(
            self.num_subnormals
                .expect("Should have called setup_inputs first"),
            &mut self.input_storage,
            rng,
            inside_test,
            multiplier,
        );

        // Set up benchmark run
        FmaAddendRun {
            inputs: self.input_storage.freeze(),
            accumulators: operations::narrow_accumulators(rng, inside_test),
            multiplier,
            inv_multiplier,
        }
    }

    type Run<'run>
        = FmaAddendRun<Storage::Frozen<'run>, ILP>
    where
        Self: 'run;
}

/// [`BenchmarkRun`] of [`FmaAddendBenchmark`]
struct FmaAddendRun<Storage: Inputs, const ILP: usize> {
    inputs: Storage,
    accumulators: [Storage::Element; ILP],
    multiplier: Storage::Element,
    inv_multiplier: Storage::Element,
}
//
impl<Storage: Inputs, const ILP: usize> BenchmarkRun for FmaAddendRun<Storage, ILP> {
    type Float = Storage::Element;

    fn inputs(&self) -> &[Self::Float] {
        self.inputs.as_ref()
    }

    #[inline]
    fn integrate_inputs(&mut self) {
        let multiplier = self.multiplier;
        let inv_multiplier = self.inv_multiplier;
        operations::integrate_pairs(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, true>,
            &self.inputs,
            move |acc, [elem1, elem2]| {
                // - By flipping between a multiplier and its inverse, we ensure
                //   that the accumulator keeps the same order of magnitude over
                //   time.
                // - Normal input element generation is biased such that after
                //   adding a quantity to the accumulator, the same quantity is
                //   later subtracted back from it, with correct accounting of
                //   the effect of past multipliers.
                acc.mul_add(multiplier, elem1)
                    .mul_add(inv_multiplier, elem2)
            },
        );
    }

    #[inline]
    fn accumulators(&self) -> &[Self::Float] {
        &self.accumulators
    }
}

/// Generate an input dataset
fn generate_inputs<Storage: InputsMut, const ILP: usize>(
    num_subnormals: usize,
    input_storage: &mut Storage,
    rng: &mut impl Rng,
    inside_test: bool,
    multiplier: Storage::Element,
) {
    generate_input_pairs::<_, _, ILP>(
        num_subnormals,
        input_storage,
        rng,
        inside_test,
        FmaAddendGenerator::new(multiplier),
    )
}

/// Global state of the input generator
struct FmaAddendGenerator<T: FloatLike> {
    /// Initial multiplier of the FMA cycle
    multiplier: T,

    /// Inverse of `Self::multiplier`
    inv_multiplier: T,
}
//
impl<T: FloatLike> FmaAddendGenerator<T> {
    /// Set up an input generator
    fn new(multiplier: T) -> Self {
        let inv_multiplier = T::splat(1.0) / multiplier;
        debug_assert_eq!(T::splat(1.0) / inv_multiplier, multiplier);
        Self {
            multiplier,
            inv_multiplier,
        }
    }
}
//
impl<T: FloatLike, R: Rng> InputGenerator<T, R> for FmaAddendGenerator<T> {
    type Stream<'a> = FmaAddendStream<'a, T>;

    fn new_stream(&self) -> Self::Stream<'_> {
        FmaAddendStream {
            generator: self,
            next_multiplier: Multiplier::Direct,
            state: FmaAddendState::Unconstrained,
        }
    }
}

/// Per-stream state of the input generator
struct FmaAddendStream<'generator, T: FloatLike> {
    /// Back-reference to the common generator state
    generator: &'generator FmaAddendGenerator<T>,

    /// Next multiplier for this stream
    next_multiplier: Multiplier,

    /// Per-stream state machine
    state: FmaAddendState<T>,
}
//
/// Inner state machine of [`FmaAddendStream`]
#[derive(Clone, Debug, PartialEq)]
enum FmaAddendState<T: FloatLike> {
    /// No constraint on the next normal value
    Unconstrained,

    /// Next normal value should cancel out the accumulator change caused by the
    /// previously added normal input value
    ScaledNegate {
        /// Value that was previously added
        value: T,

        /// `global_idx` at the time where `value` was added
        global_idx: usize,

        /// `self.next_multiplier` at the time where `value` was added
        ///
        /// After a value is added to the accumulator, the accumulator will be
        /// repeatedly multiplied by an alternating sequence of `multiplier` and
        /// `inv_multiplier` along with the rest of the accumulator.
        ///
        /// If we denote `acc0` the original accumulator, then depending on what
        /// the starting point of this sequence is, the accumulator will either
        /// become `(acc0 + value) * multiplier` or `(acc0 + value) *
        /// inv_multiplier` on the next iteration, then go back to `acc0 +
        /// value` by virtue of being multiplied by the inverse of the previous
        /// multiplier, and subsequently keep flipping between these two values.
        ///
        /// When the time comes to cancel out the previously added `value`, if
        /// `self.next_multipler` is the same as it was at the point where
        /// `value` was added, then injecting `-value` in the stream is enough
        /// to cancel out the effect of the previous addend. Otherwise we must
        /// inject `-value * multiplier` or `-value / multiplier` to compensate
        /// for the factor that was applied to the previously added value.
        next_multiplier: Multiplier,
    },
}
//
/// What the next FMA operation is going to multiply the accumulator by
///
/// The FMA operation after that is going to cancel the effect out by
/// multiplying the accumulator by the inverse of that number, then the next FMA
/// operation will perform the same multiplication again, and so on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Multiplier {
    /// Multiply by `FmaAddendGenerator::multiplier`
    Direct,

    /// Multiply by `FmaAddendGenerator::inv_multiplier`
    Inverse,
}
//
impl Multiplier {
    fn flip(&mut self) {
        *self = !*self;
    }
}
//
impl Not for Multiplier {
    type Output = Self;
    fn not(self) -> Self {
        match self {
            Self::Direct => Self::Inverse,
            Self::Inverse => Self::Direct,
        }
    }
}
//
impl<T: FloatLike, R: Rng> GeneratorStream<R> for FmaAddendStream<'_, T> {
    type Float = T;

    #[inline]
    fn record_subnormal(&mut self) {
        // Adding a subnormal number to an accumulator of order of magnitude ~1
        // has no effect, so the only effect of integrating a subnormal is that
        // the accumulator will be multiplied by `multiplier` or
        // `inv_multiplier`.
        self.next_multiplier.flip();
    }

    #[inline]
    fn generate_normal(
        &mut self,
        global_idx: usize,
        rng: &mut R,
        mut narrow: impl FnMut(&mut R) -> T,
    ) -> T {
        // Update multiplier first because the new addend is added after the
        // current accumulator has been multiplied by the current multiplier.
        self.next_multiplier.flip();
        match self.state {
            FmaAddendState::Unconstrained => {
                let value = narrow(rng);
                self.state = FmaAddendState::ScaledNegate {
                    value,
                    global_idx,
                    next_multiplier: self.next_multiplier,
                };
                value
            }
            FmaAddendState::ScaledNegate {
                value,
                next_multiplier,
                ..
            } => {
                self.state = FmaAddendState::Unconstrained;
                let mut inverse = -value;
                if next_multiplier != self.next_multiplier {
                    let factor = match next_multiplier {
                        Multiplier::Direct => self.generator.multiplier,
                        Multiplier::Inverse => self.generator.inv_multiplier,
                    };
                    inverse *= factor;
                }
                inverse
            }
        }
    }

    fn finalize(self, mut stream: DataStream<'_, T>, _rng: &mut R) {
        // The last added normal value cannot be canceled out, so make it zero
        // to enforce that the accumulator is back to its initial value at the
        // end of the benchmark run.
        if let FmaAddendState::ScaledNegate { global_idx, .. } = self.state {
            *stream.scalar_at(global_idx) = T::splat(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{
        floats::{self, test_utils::suggested_extremal_bias},
        inputs::generators::test_utils::{
            num_normals_subnormals, stream_target_subnormals, target_and_num_subnormals,
        },
        operations::test_utils::NeedsNarrowAcc,
        test_utils::assert_panics,
    };
    use itertools::Itertools;
    use proptest::prelude::*;
    use rand::rngs::ThreadRng;
    use std::panic::AssertUnwindSafe;

    /// Multiplier generator
    fn multiplier() -> impl Strategy<Value = f32> {
        prop_oneof![Just(0.5), Just(2.0),]
    }

    proptest! {
        /// Test [`FmaAddendGenerator`] and [`FmaAddendStream`]
        #[test]
        fn fma_addend_generator(
            (stream_idx, num_streams, mut target, subnormals) in stream_target_subnormals(true),
            multiplier in multiplier(),
        ) {
            // Set up an input generator
            let generator = FmaAddendGenerator::new(multiplier);
            prop_assert_eq!(generator.multiplier, multiplier);
            prop_assert_eq!(generator.inv_multiplier, 1.0 / multiplier);
            prop_assert_eq!(1.0 / generator.inv_multiplier, multiplier);

            // Set up a mock data stream
            let rng = &mut rand::thread_rng();
            let [num_narrow, num_subnormal] = num_normals_subnormals(&subnormals);
            let mut narrow = floats::narrow_sampler(suggested_extremal_bias(num_narrow));
            let subnormal = floats::subnormal_sampler(suggested_extremal_bias(num_subnormal));
            let mut stream = <FmaAddendGenerator<f32> as InputGenerator<f32, ThreadRng>>::new_stream(&generator);
            prop_assert!(std::ptr::eq(stream.generator, &generator));
            prop_assert_eq!(stream.next_multiplier, Multiplier::Direct);
            prop_assert_eq!(&stream.state, &FmaAddendState::Unconstrained);

            // Simulate the creation of a dataset, checking behavior
            debug_assert_eq!(target.len() % 2, 0);
            let half_len = target.len() / 2;
            let stream_indices =
                ((0..half_len).skip(stream_idx).step_by(num_streams))
                    .interleave((half_len..target.len()).skip(stream_idx).step_by(num_streams));
            for (elem_idx, is_subnormal) in stream_indices.clone().zip(subnormals) {
                let initial_state = stream.state.clone();
                let initial_multiplier = stream.next_multiplier;
                let expected_next_multiplier = !initial_multiplier;
                let new_value = if is_subnormal {
                    <FmaAddendStream<_> as GeneratorStream<ThreadRng>>::record_subnormal(&mut stream);
                    prop_assert_eq!(&stream.state, &initial_state);
                    subnormal(rng)
                } else {
                    let output: f32 = stream.generate_normal(elem_idx, rng, &mut narrow);
                    match (initial_state, &stream.state) {
                        (FmaAddendState::Unconstrained, FmaAddendState::ScaledNegate { global_idx, value, next_multiplier }) => {
                            prop_assert_eq!(*global_idx, elem_idx);
                            prop_assert!(*value >= 0.5);
                            prop_assert!(*value < 2.0);
                            prop_assert_eq!(output, *value);
                            prop_assert_eq!(*next_multiplier, expected_next_multiplier);
                        }
                        (FmaAddendState::ScaledNegate { global_idx: _, value, next_multiplier }, FmaAddendState::Unconstrained) => {
                            let mut expected = -value;
                            if next_multiplier != expected_next_multiplier {
                                match next_multiplier {
                                    Multiplier::Direct => expected *= multiplier,
                                    Multiplier::Inverse => expected /= multiplier,
                                }
                            }
                            prop_assert_eq!(output, expected);
                        }
                        (from, to) => {
                            return Err(TestCaseError::fail(
                                format!("Did not expect a stream state transition from {from:?} to {to:?}")
                            ));
                        }
                    }
                    output
                };
                prop_assert_eq!(stream.next_multiplier, expected_next_multiplier);
                target[elem_idx] = new_value;
            }

            // Determine the expected final output
            let mut expected = target.clone();
            match &stream.state {
                FmaAddendState::Unconstrained => {}
                FmaAddendState::ScaledNegate { global_idx, .. } => expected[*global_idx] = 0.0,
            };

            // Check that this is actually the output we get
            <FmaAddendStream<_> as GeneratorStream<ThreadRng>>::finalize(
                stream,
                DataStream::new(&mut target, stream_idx, num_streams),
                rng,
            );
            prop_assert!(
                target.into_iter().map(f32::to_bits)
                .eq(expected.into_iter().map(f32::to_bits))
            );
        }
    }

    /// Test [`generate_inputs()`]
    fn test_generate_inputs<const ILP: usize>(
        mut target: &mut [f32],
        num_subnormals: usize,
        multiplier: f32,
    ) -> Result<(), TestCaseError> {
        // Generate the inputs
        let mut rng = rand::thread_rng();
        let target_len = target.len();
        if num_subnormals > target_len || target_len % 2 != 0 {
            return assert_panics(AssertUnwindSafe(|| {
                generate_inputs::<_, ILP>(num_subnormals, &mut target, &mut rng, true, multiplier);
            }));
        }
        generate_inputs::<_, ILP>(num_subnormals, &mut target, &mut rng, true, multiplier);

        // Set up narrow accumulators
        let accs_init = operations::narrow_accumulators::<f32, ILP>(&mut rng, true);

        // Split the input for pairwise iteration
        let (left, right) = target.split_at(target_len / 2);
        let left_chunks = left.chunks(ILP);
        let right_chunks = right.chunks(ILP);

        // Check the result and simulate its effect on narrow accumulators
        let mut actual_subnormals = 0;
        let mut accs = accs_init;
        let mut to_negate: [Option<(f32, Multiplier)>; ILP] = [None; ILP];
        let mut last_normal = [false; ILP];
        let error_context = |chunk_idx: usize, acc_idx| {
            fn acc_inputs<const ILP: usize>(
                half: &[f32],
                chunk_idx: usize,
                acc_idx: usize,
            ) -> impl Iterator<Item = f32> + '_ {
                half.iter()
                    .copied()
                    .skip(acc_idx)
                    .step_by(ILP)
                    .take(chunk_idx)
            }
            format!(
                "\n\
                * Starting from initial accumulator {:?} and multiplier {:?}\n\
                * After integrating input(s) {:?}\n",
                accs_init[acc_idx],
                multiplier,
                acc_inputs::<ILP>(left, chunk_idx, acc_idx)
                    .zip(acc_inputs::<ILP>(right, chunk_idx, acc_idx))
                    .collect::<Vec<_>>()
            )
        };
        for (chunk_idx, (left_chunk, right_chunk)) in left_chunks.zip(right_chunks).enumerate() {
            for (((&left_elem, &right_elem), (acc_idx, acc)), (to_negate, last_normal)) in
                (left_chunk.iter().zip(right_chunk.iter()))
                    .zip(accs.iter_mut().enumerate())
                    .zip(to_negate.iter_mut().zip(&mut last_normal))
            {
                let acc_before = *acc;
                let acc_after_left = acc_before.mul_add(multiplier, left_elem);
                let acc_after_right = acc_after_left.mul_add(1.0 / multiplier, right_elem);
                let error_context = || {
                    format!(
                        "{}* While integrating next inputs ({left_elem}, {right_elem}) into accumulator {acc_before}, with results {acc_after_left} -> {acc_after_right}\n",
                        error_context(chunk_idx, acc_idx)
                    )
                };
                for (elem, curr_multiplier) in [
                    (left_elem, Multiplier::Direct),
                    (right_elem, Multiplier::Inverse),
                ] {
                    let next_multiplier = !curr_multiplier;
                    if elem.is_subnormal() {
                        actual_subnormals += 1;
                    } else if elem == 0.0 {
                        prop_assert!(to_negate.is_none(), "{}", error_context());
                        *last_normal = true;
                    } else {
                        prop_assert!(elem.is_normal(), "{}", error_context());
                        prop_assert!(!*last_normal, "{}", error_context());
                        if let Some((negated, negated_multiplier)) = to_negate {
                            let mut expected = -*negated;
                            if *negated_multiplier != next_multiplier {
                                match negated_multiplier {
                                    Multiplier::Direct => expected *= multiplier,
                                    Multiplier::Inverse => expected /= multiplier,
                                }
                            }
                            prop_assert_eq!(
                                elem,
                                expected,
                                "{}* Expected input {} that cancels out the previous input {}, but got {}",
                                error_context(),
                                expected,
                                negated,
                                elem
                            );
                            *to_negate = None;
                        } else {
                            prop_assert!(
                                (0.5..=2.0).contains(&elem),
                                "{}* Expected an input in narrow range [0.5; 2] but got {elem}",
                                error_context()
                            );
                            *to_negate = Some((elem, next_multiplier))
                        }
                    }
                    let curr_multiplier = match curr_multiplier {
                        Multiplier::Direct => multiplier,
                        Multiplier::Inverse => 1.0 / multiplier,
                    };
                    *acc = acc.mul_add(curr_multiplier, elem);
                }
            }
        }

        // Check that the input meets its goals of subnormal count and acc preservation
        prop_assert_eq!(actual_subnormals, num_subnormals);
        for (acc, init) in accs.iter().zip(accs_init) {
            let difference = (acc - init).abs();
            let threshold = 30.0 * f32::EPSILON * init;
            prop_assert!(
                difference < threshold,
                "{acc} is too far from {init} (difference {difference} above threshold {threshold} after integrating inputs {target:?}"
            )
        }
        Ok(())
    }
    //
    proptest! {
        #[test]
        fn generate_inputs_ilp1(
            (mut target, num_subnormals) in target_and_num_subnormals(1),
            multiplier in multiplier()
        ) {
            test_generate_inputs::<1>(&mut target, num_subnormals, multiplier)?;
        }

        #[test]
        fn generate_inputs_ilp2(
            (mut target, num_subnormals) in target_and_num_subnormals(2),
            multiplier in multiplier()
        ) {
            test_generate_inputs::<2>(&mut target, num_subnormals, multiplier)?;
        }

        #[test]
        fn generate_add_inputs_ilp3(
            (mut target, num_subnormals) in target_and_num_subnormals(3),
            multiplier in multiplier()
        ) {
            test_generate_inputs::<3>(&mut target, num_subnormals, multiplier)?;
        }
    }

    // Test the `Operation` implementation
    common::test_pairwise_operation!(FmaAddend, NeedsNarrowAcc::Always, 1, 30.0 * f32::EPSILON);
}
