use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::{self, FloatLike},
    inputs::{
        generators::{generate_input_pairs, DataStream, GeneratorStream, InputGenerator},
        Inputs, InputsMut,
    },
    operations::{self, Benchmark, BenchmarkRun, Operation},
};
use rand::Rng;

/// FMA with possibly subnormal inputs, followed by MAX
#[derive(Clone, Copy)]
pub struct FmaFullMax;
//
impl Operation for FmaFullMax {
    const NAME: &str = "fma_full_max";

    // One register for lower bound
    fn aux_registers_regop(_input_registers: usize) -> usize {
        1
    }

    // Initial accumulator value is not reused after FMA, so can use one memory
    // operand. But no known hardware supports two memory operands for FMA, so
    // we still need to reserve one register for loading the other input.
    const AUX_REGISTERS_MEMOP: usize = 2 + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<Storage: InputsMut, const ILP: usize>(
        input_storage: Storage,
    ) -> impl Benchmark<Float = Storage::Element> {
        assert_eq!(
            input_storage.as_ref().len() % 2,
            0,
            "Invalid test input length"
        );
        FmaFullMaxBenchmark::<_, ILP> {
            input_storage,
            num_subnormals: None,
        }
    }
}

/// [`Benchmark`] of [`FmaFullMax`]
struct FmaFullMaxBenchmark<Storage: InputsMut, const ILP: usize> {
    input_storage: Storage,
    num_subnormals: Option<usize>,
}
//
impl<Storage: InputsMut, const ILP: usize> Benchmark for FmaFullMaxBenchmark<Storage, ILP> {
    type Float = Storage::Element;

    fn num_operations(&self) -> usize {
        operations::accumulated_len(&self.input_storage, ILP) / 2
    }

    fn setup_inputs(&mut self, num_subnormals: usize) {
        assert!(num_subnormals <= self.input_storage.as_ref().len());
        self.num_subnormals = Some(num_subnormals);
    }

    #[inline]
    fn start_run(&mut self, rng: &mut impl Rng) -> Self::Run<'_> {
        let accumulators = operations::narrow_accumulators(rng);
        generate_input_pairs::<_, _, ILP>(
            &mut self.input_storage,
            rng,
            self.num_subnormals
                .expect("Should have called setup_inputs first"),
            FmaFullMaxGenerator,
        );
        FmaFullMaxRun {
            inputs: self.input_storage.freeze(),
            accumulators,
        }
    }

    type Run<'run>
        = FmaFullMaxRun<Storage::Frozen<'run>, ILP>
    where
        Self: 'run;
}

/// [`BenchmarkRun`] of [`FmaFullMaxBenchmark`]
struct FmaFullMaxRun<Storage: Inputs, const ILP: usize> {
    inputs: Storage,
    accumulators: [Storage::Element; ILP],
}
//
impl<Storage: Inputs, const ILP: usize> BenchmarkRun for FmaFullMaxRun<Storage, ILP> {
    type Float = Storage::Element;

    fn inputs(&self) -> &[Self::Float] {
        self.inputs.as_ref()
    }

    #[inline]
    fn integrate_inputs(&mut self) {
        let lower_bound = lower_bound::<Storage::Element>();
        operations::integrate_pairs(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, false>,
            &self.inputs,
            move |acc, [elem1, elem2]| {
                // The input data stream is generated using a rather elaborate
                // state machine to ensure that the accumulator stays as close
                // as possible to the "narrow" [1/2; 2[ range that it initially
                // belongs to, for the entire duration of the benchmark.
                //
                // See `FmaFullMaxState` below for a detailed description of the
                // various states that the accumulator can end up in, and the
                // way it gets back to the `Narrow` state from there.
                operations::hide_single_accumulator(acc.mul_add(elem1, elem2)).fast_max(lower_bound)
            },
        );
    }

    #[inline]
    fn accumulators(&self) -> &[Self::Float] {
        &self.accumulators
    }
}

/// Lower bound that is imposed on the accumulator value
fn lower_bound<T: FloatLike>() -> T {
    T::splat(0.25)
}

/// Global state of the input generator
struct FmaFullMaxGenerator;
//
impl<F: FloatLike, R: Rng> InputGenerator<F, R> for FmaFullMaxGenerator {
    type Stream<'a>
        = FmaFullMaxStream<F>
    where
        Self: 'a;

    fn new_stream(&self) -> Self::Stream<'_> {
        FmaFullMaxStream {
            next_role: ValueRole::Multiplier,
            state: FmaFullMaxState::Narrow,
        }
    }
}

/// Per-stream state of the input generator
struct FmaFullMaxStream<T: FloatLike> {
    /// Role of the next number that this stream is going to emit
    next_role: ValueRole,

    /// Per-stream state machine
    state: FmaFullMaxState<T>,
}
//
/// What is the role of the next value emitted by this stream
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ValueRole {
    Multiplier,
    Addend,
}
//
impl ValueRole {
    fn flip(&mut self) {
        *self = match *self {
            Self::Multiplier => Self::Addend,
            Self::Addend => Self::Multiplier,
        }
    }
}
//
/// Inner state machine of [`FmaFullMaxStream`]
#[derive(Clone, Copy, Debug, PartialEq)]
enum FmaFullMaxState<T: FloatLike> {
    /// Accumulator is a normal number in range `[1/2; 2[`
    ///
    /// This is the initial state of the accumulator. Every multiplier or addend
    /// in the normal range strives to bring the accumulator back to
    /// this range whenever it deviates from it.
    ///
    /// Possible state transitions:
    ///
    /// - If the FMA starts from this state...
    ///   * Multiplying by a normal number leads to the `NarrowProduct` state
    ///   * Multiplying by a subnormal number leads to the `Subnormal` state
    /// - If this state is reached by multiplying a `NarrowProduct` accumulator
    ///   by a normal multiplier...
    ///   * Adding a normal number leads to the `NarrowSum` state
    ///   * Adding a subnormal number preserves the `Narrow` state
    Narrow,

    /// Accumulator is the product of two normal numbers in range `[1/2; 2[`,
    /// which is in range `[1/4; 4[`
    ///
    /// This state is reached by multiplying a `Narrow` accumulator by a normal
    /// input. After this, the following state transitions can occur:
    ///
    /// - If this state is reached by multiplying a `Narrow` accumulator...
    ///   * Normal addends are forced to be 0.0 for precision reasons, which
    ///     preserves the `NarrowProduct` state.
    ///   * Adding a subnormal number also preserves the `NarrowProduct` state
    /// - If the FMA starts from this state...
    ///   * Multiplying by a normal number leads to the `Narrow` state
    ///   * Multiplying by a subnormal number leads to the `Subnormal` state
    NarrowProduct {
        /// Position of the factor in the input data stream
        global_idx: usize,

        /// Factor that was applied
        factor: T,
    },

    /// Accumulator is the sum of two normal numbers in range `[1/2; 2[`, which
    /// is in range `[1; 4[`
    ///
    /// This state is reached by reaching the `Narrow` state via a normal
    /// multiplier, then adding a normal number. After this, the following state
    /// transitions can occur:
    ///
    /// - Normal multipliers are forced to be 1.0 for precision reasons, which
    ///   preserves the `NarrowSum` state. After this...
    ///     * A normal addend takes us back to the `Narrow` state
    ///     * A subnormal addend preserves the `NarrowSum` state
    /// - Multiplying by a subnormal number leads to the `Subnormal` state
    NarrowSum {
        /// Position of the addend in the input data stream
        global_idx: usize,

        /// Addend that was applied
        addend: T,
    },

    /// Accumulator is a subnormal number
    ///
    /// This state is reached by multiplying any number by a subnormal number.
    /// After this, the following state transitions can occur:
    ///
    /// - Adding a normal number leads back to the `Narrow` state
    /// - Adding a subnormal number leads to the MAX kicking in and clamping the
    ///   accumulator back to its `LowerBound`.
    ///
    /// This state can only be reached via multiplication, and all additive
    /// state transitions lead away from it, therefore there is no situation in
    /// which we will multiply such an accumulator by a number.
    Subnormal,

    /// Accumulator has been clamped to the lower bound `1/4`.
    ///
    /// This state is reached by multiplying an accumulator by a subnormal
    /// number, then adding another subnormal number to it. Normally, the result
    /// would be subnormal, but we use a MAX to clamp the accumulator back to
    /// the lower bound so that it stays normal across future operations.
    ///
    /// After this, the following state transitions can occur:
    ///
    /// - Multiplying by a normal number leads back to the `Narrow` state
    /// - Multiplying by a subnormal number leads to the `Subnormal` state
    ///
    /// This state can only be reached via addition, and all multiplicative
    /// state transitions lead away from it, therefore there is no situation in
    /// which we will add a number to such an accumulator.
    LowerBound,
}
//
impl<T: FloatLike, R: Rng> GeneratorStream<R> for FmaFullMaxStream<T> {
    type Float = T;

    #[inline]
    fn record_subnormal(&mut self) {
        self.state = match self.next_role {
            ValueRole::Multiplier => match self.state {
                FmaFullMaxState::Narrow
                | FmaFullMaxState::NarrowProduct { .. }
                | FmaFullMaxState::NarrowSum { .. }
                | FmaFullMaxState::LowerBound => FmaFullMaxState::Subnormal,
                FmaFullMaxState::Subnormal => {
                    unreachable!("Previous MAX should have taken us out of this state")
                }
            },
            ValueRole::Addend => match self.state {
                normal @ (FmaFullMaxState::Narrow
                | FmaFullMaxState::NarrowProduct { .. }
                | FmaFullMaxState::NarrowSum { .. }) => normal,
                FmaFullMaxState::Subnormal => FmaFullMaxState::LowerBound,
                FmaFullMaxState::LowerBound => {
                    unreachable!("Previous multiplier should have taken us out of these states")
                }
            },
        };
        self.next_role.flip();
    }

    #[inline]
    fn generate_normal(
        &mut self,
        global_idx: usize,
        rng: &mut R,
        mut narrow: impl FnMut(&mut R) -> T,
    ) -> T {
        let (value, next_state) = match self.next_role {
            ValueRole::Multiplier => match self.state {
                FmaFullMaxState::Narrow => {
                    let factor = narrow(rng);
                    (
                        factor,
                        FmaFullMaxState::NarrowProduct { global_idx, factor },
                    )
                }
                FmaFullMaxState::NarrowProduct {
                    global_idx: _,
                    factor,
                } => {
                    // accumulator * factor * 1/factor ~ accumulator
                    (T::splat(1.0) / factor, FmaFullMaxState::Narrow)
                }
                FmaFullMaxState::NarrowSum { .. } => (T::splat(1.0), self.state),
                FmaFullMaxState::LowerBound => {
                    let recovery_multiplier = lower_recovery_multiplier(rng);
                    (recovery_multiplier, FmaFullMaxState::Narrow)
                }
                FmaFullMaxState::Subnormal => {
                    unreachable!("Previous MAX should have taken us out of this state")
                }
            },
            ValueRole::Addend => {
                match self.state {
                    FmaFullMaxState::Narrow => {
                        let addend = narrow(rng);
                        (addend, FmaFullMaxState::NarrowSum { global_idx, addend })
                    }
                    FmaFullMaxState::NarrowProduct { .. } => (T::splat(0.0), self.state),
                    FmaFullMaxState::NarrowSum {
                        global_idx: _,
                        addend,
                    } => {
                        // accumulator + addend - addend = accumulator
                        (-addend, FmaFullMaxState::Narrow)
                    }
                    FmaFullMaxState::Subnormal => {
                        // ~0 + narrow ~ narrow
                        let narrow = narrow(rng);
                        (narrow, FmaFullMaxState::Narrow)
                    }
                    FmaFullMaxState::LowerBound => {
                        unreachable!("Previous multiplier should have taken us out of these states")
                    }
                }
            }
        };
        self.state = next_state;
        self.next_role.flip();
        value
    }

    fn finalize(self, mut stream: DataStream<'_, T>, rng: &mut R) {
        assert_eq!(self.next_role, ValueRole::Multiplier);
        match self.state {
            FmaFullMaxState::Narrow => {}
            FmaFullMaxState::NarrowProduct { global_idx, .. } => {
                // This accumulator change cannot be compensated by a subsequent
                // input, so make sure the previous product does not actually
                // change the value of the accumulator.
                *stream.scalar_at(global_idx) = T::splat(1.0);
            }
            FmaFullMaxState::NarrowSum { global_idx, .. } => {
                // Same idea, but with sums rather than product so the neutral
                // element becomes 0.0 instead of 1.0.
                *stream.scalar_at(global_idx) = T::splat(0.0);
            }
            FmaFullMaxState::LowerBound => {
                // The accumulator ends at an abnormal magnitude. This can only
                // be tolerated if the the first multiplier of the stream is
                // subnormal, which causes this abnormal magnitude to be ignored
                // by the next benchmark run (which will start by ~zeroing out
                // the accumulator).
                let mut pairs = stream.into_pair_iter();
                let [first_multiplier, first_addend] = pairs
                    .next()
                    .expect("LowerBound is only reachable after >= 1 all-subnormal input pair");
                if first_multiplier.is_subnormal() {
                    return;
                }
                assert!(first_multiplier.is_normal());

                // Otherwise we should extract the last subnormal
                // (addend, multiplier) pair...
                let [last_multiplier, last_addend] = pairs.next_back().expect(
                    "Last input pair should be all-subnormal, it cannot be the first input pair if its multiplier is normal",
                );
                assert!(last_multiplier.is_subnormal());
                assert!(last_addend.is_subnormal());

                // ...and swap it with the first pair coming before it in the
                // sequence that is not all-subnormal. This is fine because...
                // - There has to be a previous input pair that is not
                //   all-subnormal because we just found the first pair isn't.
                // - Removing a normal input that comes before a sequence of
                //   all-subnormal values does not meaningfully affect
                //   accumulator magnitude since the all-subnormal values
                //   destroy the previous accumulator magnitude anyway.
                let [prev_multiplier, prev_addend] = pairs
                    .rev()
                    .chain(std::iter::once([first_multiplier, first_addend]))
                    .find(|[mul, add]| mul.is_normal() || add.is_normal())
                    .expect("At least true for [first_multiplier, first_addend]");
                std::mem::swap(last_multiplier, prev_multiplier);
                std::mem::swap(last_addend, prev_addend);

                // Finally, we should fix up the new last [multiplier, addend]
                // pair so that it brings the accumulator back to the "narrow"
                // [1/2; 2[ magnitude range that it started from.
                match (last_multiplier.is_normal(), last_addend.is_normal()) {
                    (true, true) | (true, false) => {
                        *last_multiplier = lower_recovery_multiplier(rng);
                        if last_addend.is_normal() {
                            *last_addend = T::splat(0.0);
                        }
                    }
                    (false, true) => {
                        let narrow = floats::narrow_sampler();
                        *last_addend = narrow(rng);
                    }
                    (false, false) => unreachable!(
                        "We just ensured that the last input pair is not all-subnormal"
                    ),
                };
            }
            FmaFullMaxState::Subnormal => {
                unreachable!("Previous MAX should have taken us out of this state")
            }
        }
    }
}

/// Generate a multiplier that recovers an accumulator from a subnormal value
/// (which takes it to its lower bound) back to narrow magnitude range [1/2; 2[
fn lower_recovery_multiplier<T: FloatLike>(rng: &mut impl Rng) -> T {
    // 1/4 * random(2..8) = random(1/2..2)
    assert_eq!(lower_bound::<T>(), T::splat(0.25));
    T::sampler(1..3)(rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::operations::test_utils::NeedsNarrowAcc;
    common::test_pairwise_operation!(FmaFullMax, NeedsNarrowAcc::FirstNormal, 2);
}
