use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
    operation::{self, Benchmark, Operation},
};
use rand::Rng;

/// MUL followed by averaging
#[derive(Clone, Copy)]
pub struct MulAverage;
//
impl<T: FloatLike> Operation<T> for MulAverage {
    const NAME: &str = "mul_average";

    // One register for the averaging weight, another for the averaging target
    const AUX_REGISTERS_REGOP: usize = 2;

    // Inputs are directly reduced into the accumulator, can use memory operands
    const AUX_REGISTERS_MEMOP: usize = 2 + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        MulAverageBenchmark {
            accumulators: [Default::default(); ILP],
            target: Default::default(),
        }
    }
}

/// [`Benchmark`] of [`MulAverage`]
#[derive(Clone, Copy)]
struct MulAverageBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
    target: T,
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for MulAverageBenchmark<T, ILP> {
    type Float = T;

    fn num_operations<Inputs: FloatSet>(inputs: &Inputs) -> usize {
        inputs.reused_len(ILP)
    }

    fn begin_run(&mut self, mut rng: impl Rng) {
        let normal_sampler = T::normal_sampler();
        self.accumulators = operation::multiplicative_accumulators(&mut rng);
        self.target = normal_sampler(&mut rng);
    }

    #[inline]
    fn integrate_inputs<Inputs>(mut self, inputs: Inputs) -> (Self, Inputs)
    where
        Inputs: FloatSequence<Element = Self::Float>,
    {
        let (next_accs, next_inputs) =
            operation::integrate_full(self.accumulators, inputs, move |acc, elem| {
                (acc * elem + self.target) * T::splat(0.5)
            });
        self.accumulators = next_accs;
        (self, next_inputs)
    }

    fn consume_outputs(self) {
        operation::consume_accumulators(self.accumulators);
    }
}
