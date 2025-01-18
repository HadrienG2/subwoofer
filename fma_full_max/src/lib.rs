use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{self, DataStream, GeneratorStream, InputGenerator, Inputs, InputsMut},
    operations::{self, Benchmark, BenchmarkRun, Operation},
};
use rand::Rng;
use std::cell::RefCell;

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

    fn make_benchmark<const ILP: usize>(input_storage: impl InputsMut) -> impl Benchmark {
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
    fn num_operations(&self) -> usize {
        operations::accumulated_len(&self.input_storage, ILP) / 2
    }

    fn setup_inputs(&mut self, num_subnormals: usize) {
        self.num_subnormals = Some(num_subnormals);
    }

    #[inline]
    fn start_run(&mut self, rng: &mut impl Rng) -> Self::Run<'_> {
        let accumulators = operations::narrow_accumulators(rng);
        inputs::generate_input_pairs::<_, _, ILP>(
            &mut self.input_storage,
            rng,
            self.num_subnormals
                .expect("Should have called setup_inputs first"),
            FmaFullMaxGenerator(RefCell::new(accumulators.into_iter())),
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
                // various states that the accumulator that end up in, and the
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
///
/// Will produce one valid data stream per initial accumulator value. Subsequent
/// data streams are invalid and only suitable for use as placeholder values,
/// they should not be manipulated.
struct FmaFullMaxGenerator<AccIter: Iterator>(RefCell<AccIter>)
where
    AccIter::Item: FloatLike;
//
impl<AccIter: Iterator, R: Rng> InputGenerator<AccIter::Item, R> for FmaFullMaxGenerator<AccIter>
where
    AccIter::Item: FloatLike,
{
    type Stream<'a>
        = FmaFullMaxStream<AccIter::Item>
    where
        Self: 'a;

    fn new_stream(&self) -> Self::Stream<'_> {
        FmaFullMaxStream {
            next_role: ValueRole::Multiplier,
            state: self
                .0
                .borrow_mut()
                .next()
                .map_or(FmaFullMaxState::Invalid, FmaFullMaxState::Narrow),
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
    /// in the normal range strives to bring to bring the accumulator back to
    /// this range whenever it deviates from it.
    ///
    /// Possible state transitions:
    ///
    /// - If the FMA starts from this state...
    ///   * Multiplying by a normal number leads to the `NarrowProduct` state
    ///   * Multiplying by a subnormal number leads to the `Subnormal` state
    /// - If this state is reached by multiplying a non-`Narrow` accumulator by
    ///   a normal multiplier...
    ///   * Adding a normal number leads to the `NarrowSum` state
    ///   * Adding a subnormal number preserves the `Narrow` state
    Narrow(T),

    /// Accumulator is the product of two normal numbers in range `[1/2; 2[`,
    /// which is in range `[1/4; 4[`
    ///
    /// This state is reached by multiplying a `Narrow` accumulator by a normal
    /// input. After this, the following state transitions can occur:
    ///
    /// - If this state is reached by multiplying a `Narrow` accumulator...
    ///   * Adding a normal number leads back to the `Narrow` state
    ///   * Adding a subnormal number preserves the `NarrowProduct` state
    /// - If the FMA starts from this state...
    ///   * Multiplying by a normal number leads to the `Narrow` state
    ///   * Multiplying by a subnormal number leads to the `Subnormal` state
    NarrowProduct {
        /// Accumulator value before the product
        accumulator: T,

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
    /// - Multiplying by a normal number leads back to the `Narrow` state
    /// - Multiplying by a subnormal number leads to the `Subnormal` state
    ///
    /// This state can only be reached via addition, and all multiplicative
    /// state transitions lead away from it, therefore there is no situation in
    /// which we will add a number to such an accumulator.
    NarrowSum {
        /// Accumulator value before the sum
        accumulator: T,

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

    /// This state should never encountered by a valid data stream
    Invalid,
}
//
impl<T: FloatLike, R: Rng> GeneratorStream<R> for FmaFullMaxStream<T> {
    type Float = T;

    #[inline]
    fn record_subnormal(&mut self) {
        self.state = match self.next_role {
            ValueRole::Multiplier => match self.state {
                FmaFullMaxState::Narrow(_)
                | FmaFullMaxState::NarrowProduct { .. }
                | FmaFullMaxState::NarrowSum { .. }
                | FmaFullMaxState::LowerBound => FmaFullMaxState::Subnormal,
                FmaFullMaxState::Subnormal => {
                    unreachable!("Previous MAX should have taken us out of this state")
                }
                FmaFullMaxState::Invalid => {
                    unreachable!("Attempted to use an invalid input data stream")
                }
            },
            ValueRole::Addend => match self.state {
                normal @ (FmaFullMaxState::Narrow(_) | FmaFullMaxState::NarrowProduct { .. }) => {
                    normal
                }
                FmaFullMaxState::Subnormal => FmaFullMaxState::LowerBound,
                FmaFullMaxState::NarrowSum { .. } | FmaFullMaxState::LowerBound => {
                    unreachable!("Previous multiplier should have taken us out of these states")
                }
                FmaFullMaxState::Invalid => {
                    unreachable!("Attempted to use an invalid input data stream")
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
                FmaFullMaxState::Narrow(accumulator) => {
                    let factor = narrow(rng);
                    (
                        factor,
                        FmaFullMaxState::NarrowProduct {
                            accumulator,
                            global_idx,
                            factor,
                        },
                    )
                }
                FmaFullMaxState::NarrowProduct {
                    accumulator,
                    global_idx: _,
                    factor,
                } => {
                    // accumulator * factor * 1/factor ~ accumulator
                    (T::splat(1.0) / factor, FmaFullMaxState::Narrow(accumulator))
                }
                FmaFullMaxState::NarrowSum {
                    accumulator,
                    global_idx: _,
                    addend,
                } => {
                    //   (accumulator + addend)
                    // * accumulator / (accumulator + addend)
                    // ~ accumulator
                    (
                        accumulator / (accumulator + addend),
                        FmaFullMaxState::Narrow(accumulator),
                    )
                }
                FmaFullMaxState::LowerBound => {
                    // 1/4 * random(2..8) = random(1/2..2)
                    assert_eq!(lower_bound::<T>(), T::splat(0.25));
                    let _2to8 = T::sampler(1..3)(rng);
                    (_2to8, FmaFullMaxState::Narrow(_2to8 * lower_bound::<T>()))
                }
                FmaFullMaxState::Subnormal => {
                    unreachable!("Previous MAX should have taken us out of this state")
                }
                FmaFullMaxState::Invalid => {
                    unreachable!("Attempted to use an invalid input data stream")
                }
            },
            ValueRole::Addend => {
                match self.state {
                    FmaFullMaxState::Narrow(accumulator) => {
                        let addend = narrow(rng);
                        (
                            addend,
                            FmaFullMaxState::NarrowSum {
                                accumulator,
                                global_idx,
                                addend,
                            },
                        )
                    }
                    FmaFullMaxState::NarrowProduct {
                        accumulator,
                        global_idx: _,
                        factor,
                    } => {
                        //   accumulator * factor
                        // + accumulator * (1 - factor)
                        // ~ accumulator
                        (
                            accumulator * (T::splat(1.0) - factor),
                            FmaFullMaxState::Narrow(accumulator),
                        )
                    }
                    FmaFullMaxState::Subnormal => {
                        // ~0 + narrow ~ narrow
                        let narrow = narrow(rng);
                        (narrow, FmaFullMaxState::Narrow(narrow))
                    }
                    FmaFullMaxState::NarrowSum { .. } | FmaFullMaxState::LowerBound => {
                        unreachable!("Previous multiplier should have taken us out of these states")
                    }
                    FmaFullMaxState::Invalid => {
                        unreachable!("Attempted to use an invalid input data stream")
                    }
                }
            }
        };
        self.state = next_state;
        self.next_role.flip();
        value
    }

    fn finalize(self, mut stream: DataStream<'_, T>) {
        assert_eq!(self.next_role, ValueRole::Multiplier);
        match self.state {
            FmaFullMaxState::Narrow(_) => {}
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
                let [first_multiplier, _first_addend] = pairs
                    .next()
                    .expect("LowerBound is only reachable after >= 1 all-subnormal input pair");
                if first_multiplier.is_subnormal() {
                    return;
                }
                assert!(first_multiplier.is_normal());

                // Otherwise we should exchange the last subnormal addend with
                // the first normal input.
                //
                // This is enough to bring the accumulator to the standard
                // "narrow" range again because...
                //
                // - If we reached the LowerBound state, it implies that the
                //   last multiplier is subnormal, zeroing out any previous
                //   state from the accumulator. The addend thus fully
                //   determines the final accumulator state.
                // - The former first multiplier, if normal, is guaranteed to be
                //   in the same narrow range [1/2; 2[ that accumulators should
                //   belong to. By adding it to the aforementioned subnormal
                //   product, we end up with a Narrow initial accumulator again.
                let [last_multiplier, last_addend] = pairs.next_back().expect(
                    "Last input pair should be all-subnormal, it cannot be the first input pair if its multiplier is normal",
                );
                assert!(last_multiplier.is_subnormal());
                assert!(last_addend.is_subnormal());
                std::mem::swap(first_multiplier, last_addend);
            }
            FmaFullMaxState::Subnormal => {
                unreachable!("Previous MAX should have taken us out of this state")
            }
            FmaFullMaxState::Invalid => {
                unreachable!("Attempted to use an invalid input data stream")
            }
        }
    }
}
