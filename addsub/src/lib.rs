use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
    operations::{self, Benchmark, Operation},
};
use rand::Rng;

/// ADD/SUB cycle
#[derive(Clone, Copy)]
pub struct AddSub;
//
impl<T: FloatLike> Operation<T> for AddSub {
    const NAME: &str = "addsub";

    fn aux_registers_regop(_input_registers: usize) -> usize {
        0
    }

    // Inputs are directly reduced into the accumulator, can use memory operands
    const AUX_REGISTERS_MEMOP: usize = (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        AddSubBenchmark {
            accumulators: [Default::default(); ILP],
        }
    }
}

/// [`Benchmark`] of [`AddSub`]
#[derive(Clone, Copy)]
struct AddSubBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for AddSubBenchmark<T, ILP> {
    type Float = T;

    fn num_operations<Inputs: FloatSet>(inputs: &Inputs) -> usize {
        inputs.reused_len(ILP)
    }

    #[inline]
    fn begin_run(self, rng: impl Rng) -> Self {
        // This is just a random additive walk of ~unity or subnormal step, so
        // given a high enough starting point, an initially normal accumulator
        // should stay in the normal range forever.
        Self {
            accumulators: operations::additive_accumulators(rng),
        }
    }

    #[inline]
    fn integrate_inputs<Inputs>(&mut self, inputs: &mut Inputs)
    where
        Inputs: FloatSequence<Element = Self::Float>,
    {
        // No need for input hiding here, the compiler cannot do anything
        // dangerous with the knowledge that inputs are always the same.
        operations::integrate_halves(
            &mut self.accumulators,
            inputs,
            |acc, elem| acc + elem,
            |acc, elem| acc - elem,
        );
    }

    #[inline]
    fn consume_outputs(self) {
        operations::consume_accumulators(self.accumulators);
    }
}
