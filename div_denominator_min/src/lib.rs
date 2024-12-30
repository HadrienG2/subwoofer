use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
    operations::{self, Benchmark, Operation},
};
use rand::Rng;

/// DIV with a possibly subnormal denominator, followed by MIN
#[derive(Clone, Copy)]
pub struct DivDenominatorMin;
//
impl<T: FloatLike> Operation<T> for DivDenominatorMin {
    const NAME: &str = "div_denominator_min";

    // One register for the 0.5 lower bound
    fn aux_registers_regop(_input_registers: usize) -> usize {
        1
    }

    // Inputs are directly reduced into the accumulator, can use memory operands
    const AUX_REGISTERS_MEMOP: usize = 1 + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        DivDenominatorMinBenchmark {
            accumulators: [Default::default(); ILP],
        }
    }
}

/// [`Benchmark`] of [`MulMax`]
#[derive(Clone, Copy)]
struct DivDenominatorMinBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for DivDenominatorMinBenchmark<T, ILP> {
    type Float = T;

    fn num_operations<Inputs: FloatSet>(inputs: &Inputs) -> usize {
        inputs.reused_len(ILP)
    }

    #[inline]
    fn begin_run(self, rng: impl Rng) -> Self {
        Self {
            accumulators: operations::normal_accumulators(rng),
        }
    }

    #[inline]
    fn integrate_inputs<Inputs>(&mut self, inputs: &mut Inputs)
    where
        Inputs: FloatSequence<Element = Self::Float>,
    {
        // No need to hide inputs for this benchmark, compiler can't exploit
        // knowledge of input reuse here
        operations::integrate_full::<_, _, ILP, false>(
            &mut self.accumulators,
            inputs,
            move |acc, elem| {
                // If elem is subnormal, the min takes us back to normal range,
                // otherwise this is a truncated multiplicative random walk that
                // cannot go higher than 2.0. In that case we can only use half
                // of the available exponent range but that's plenty enough.
                (acc / elem).fast_min(T::splat(2.0))
            },
        );
    }

    #[inline]
    fn consume_outputs(self) {
        operations::consume_accumulators(self.accumulators);
    }
}
