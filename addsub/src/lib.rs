use common::{
    arch::HAS_MEMORY_OPERANDS,
    inputs::{FloatSequence, FloatSet},
    process::{self, Benchmark, Operation},
    types::FloatLike,
};
use rand::Rng;

/// ADD/SUB cycle
#[derive(Clone, Copy)]
pub struct AddSub;
//
impl<T: FloatLike> Operation<T> for AddSub {
    const NAME: &str = "addsub";

    const AUX_REGISTERS_REGOP: usize = 0;

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

    fn begin_run(&mut self, rng: impl Rng) {
        // This is just a random additive walk of ~unity or subnormal step, so given a
        // high enough starting point, an initially normal accumulator should stay in
        // the normal range forever.
        self.accumulators = process::additive_accumulators(rng);
    }

    #[inline]
    fn integrate_inputs<Inputs>(mut self, inputs: Inputs) -> (Self, Inputs)
    where
        Inputs: FloatSequence<Element = Self::Float>,
    {
        // No need for input hiding here, the compiler cannot do anything dangerous
        // with the knowledge that inputs are always the same in this benchmark.
        let (next_accs, next_inputs) = process::integrate_halves::<_, _, ILP, false>(
            self.accumulators,
            inputs,
            |acc, elem| acc + elem,
            |acc, elem| acc - elem,
        );
        self.accumulators = next_accs;
        (self, next_inputs)
    }

    fn consume_outputs(self) {
        process::consume_accumulators(self.accumulators);
    }
}
