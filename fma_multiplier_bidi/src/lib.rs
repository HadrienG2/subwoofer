use common::{
    arch::{HAS_HARDWARE_NEGATED_FMA, HAS_MEMORY_OPERANDS},
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
    operations::{self, Benchmark, Operation},
};
use rand::Rng;

/// FMA with possibly subnormal multiplier, used in an ADD/SUB pattern
#[derive(Clone, Copy)]
pub struct FmaMultiplierBidi;
//
impl<T: FloatLike> Operation<T> for FmaMultiplierBidi {
    const NAME: &str = "fma_multiplier_bidi";

    // One register for the constant multiplier, and another register for a
    // negated version of that multiplier if the target CPU does not have an
    // FNMA instruction
    fn aux_registers_regop(_input_registers: usize) -> usize {
        1 + (!HAS_HARDWARE_NEGATED_FMA) as usize
    }

    // Acccumulator is not reused after FMA, so we can use memory operands if
    // available to reduce register pressure
    const AUX_REGISTERS_MEMOP: usize =
        1 + (!HAS_HARDWARE_NEGATED_FMA) as usize + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        FmaMultiplierBidiBenchmark {
            accumulators: [Default::default(); ILP],
            multiplier: Default::default(),
        }
    }
}

/// [`Benchmark`] of [`FmaMultiplierBidi`]
#[derive(Clone, Copy)]
struct FmaMultiplierBidiBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
    multiplier: T,
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for FmaMultiplierBidiBenchmark<T, ILP> {
    type Float = T;

    fn num_operations<Inputs: FloatSet>(inputs: &Inputs) -> usize {
        inputs.reused_len(ILP)
    }

    #[inline]
    fn begin_run(self, mut rng: impl Rng) -> Self {
        let normal_sampler = T::normal_sampler();
        let multiplier = normal_sampler(&mut rng);
        let accumulators = operations::additive_accumulators(rng);
        Self {
            accumulators,
            multiplier,
        }
    }

    #[inline]
    fn integrate_inputs<Inputs>(&mut self, inputs: &mut Inputs)
    where
        Inputs: FloatSequence<Element = Self::Float>,
    {
        // Overall, this is just the addsub benchmark with a step size that is
        // at most 2x smaller/larger, which fundamentally doesn't change much
        let multiplier = self.multiplier;
        operations::integrate_halves::<_, _, ILP>(
            &mut self.accumulators,
            inputs,
            move |acc, elem| elem.mul_add(multiplier, acc),
            move |acc, elem| elem.mul_add(-multiplier, acc),
        );
    }

    #[inline]
    fn consume_outputs(self) {
        operations::consume_accumulators(self.accumulators);
    }
}
