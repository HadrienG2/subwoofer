use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::{self, FloatLike},
    inputs::{self, DataStream, GeneratorStream, InputGenerator, Inputs, InputsMut},
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

    fn make_benchmark<const ILP: usize>(input_storage: impl InputsMut) -> impl Benchmark {
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
    fn num_operations(&self) -> usize {
        operations::accumulated_len(&self.input_storage, ILP)
    }

    fn setup_inputs(&mut self, num_subnormals: usize) {
        self.num_subnormals = Some(num_subnormals);
    }

    #[inline]
    fn start_run(&mut self, rng: &mut impl Rng) -> Self::Run<'_> {
        let narrow = floats::narrow_sampler();
        let multiplier = narrow(rng);
        let inv_multiplier = Storage::Element::splat(1.0) / multiplier;
        inputs::generate_input_pairs::<_, _, ILP>(
            &mut self.input_storage,
            rng,
            self.num_subnormals
                .expect("Should have called setup_inputs first"),
            FmaAddendGenerator {
                multiplier,
                inv_multiplier,
            },
        );
        FmaAddendRun {
            inputs: self.input_storage.freeze(),
            accumulators: operations::narrow_accumulators(rng),
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
                //   the effect of multipliers.
                //
                // TODO: Check need for finer-grained optimization barriers
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

/// Global input generator state for the input generator
struct FmaAddendGenerator<T: FloatLike> {
    /// "Base" multiplier of the FMA cycle
    multiplier: T,

    /// Inverse of `Self::multiplier`
    inv_multiplier: T,
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

/// Per-stream state for this input generator
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
enum FmaAddendState<T: FloatLike> {
    /// No constraint on the next normal value
    Unconstrained,

    /// Next normal value should cancel out the accumulator change caused by the
    /// previously added normal input value
    InvertNormal {
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
        /// value, and subsequently keep flipping between these two values.
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
impl<T: FloatLike> FmaAddendStream<'_, T> {
    fn update_multipler(&mut self) {
        self.next_multiplier = match self.next_multiplier {
            Multiplier::Direct => Multiplier::Inverse,
            Multiplier::Inverse => Multiplier::Direct,
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
        self.update_multipler();
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
        self.update_multipler();
        match self.state {
            FmaAddendState::Unconstrained => {
                let value = narrow(rng);
                self.state = FmaAddendState::InvertNormal {
                    value,
                    global_idx,
                    next_multiplier: self.next_multiplier,
                };
                value
            }
            FmaAddendState::InvertNormal {
                value,
                next_multiplier,
                ..
            } => {
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

    fn finalize(self, mut stream: DataStream<'_, T>) {
        // The last added normal value cannot be canceled out, so make it zero
        // to enforce that the accumulator is back to its initial value at the
        // end of the FMA cycle.
        if let FmaAddendState::InvertNormal { global_idx, .. } = self.state {
            *stream.scalar_at(global_idx) = T::splat(0.0);
        }
    }
}
