use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
    operations::{self, Benchmark, Operation},
};
use rand::Rng;

/// FMA with possibly subnormal addend, followed by averaging
#[derive(Clone, Copy)]
pub struct FmaAddendAverage;
//
impl<T: FloatLike> Operation<T> for FmaAddendAverage {
    const NAME: &str = "fma_addend_average";

    // One register for the averaging weight, another for the averaging target
    const AUX_REGISTERS_REGOP: usize = 2;

    // Inputs are directly reduced into the accumulator, can use memory operands
    const AUX_REGISTERS_MEMOP: usize = 2 + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        FmaAddendAverageBenchmark {
            accumulators: [Default::default(); ILP],
            target: Default::default(),
        }
    }
}

/// [`Benchmark`] of [`FmaAddendAverage`]
#[derive(Clone, Copy)]
struct FmaAddendAverageBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
    target: T,
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for FmaAddendAverageBenchmark<T, ILP> {
    type Float = T;

    fn num_operations<Inputs: FloatSet>(inputs: &Inputs) -> usize {
        inputs.reused_len(ILP)
    }

    #[inline]
    fn begin_run(&mut self, mut rng: impl Rng) {
        let normal_sampler = T::normal_sampler();
        self.accumulators = operations::normal_accumulators(&mut rng);
        self.target = normal_sampler(&mut rng);
    }

    #[inline]
    fn integrate_inputs<Inputs>(mut self, inputs: Inputs) -> (Self, Inputs)
    where
        Inputs: FloatSequence<Element = Self::Float>,
    {
        let halve_weight = T::splat(0.5);
        let (next_accs, next_inputs) =
            operations::integrate_full(self.accumulators, inputs, move |acc, elem| {
                (acc.mul_add(halve_weight, elem) + self.target) * halve_weight
            });
        self.accumulators = next_accs;
        (self, next_inputs)
    }

    #[inline]
    fn consume_outputs(self) {
        operations::consume_accumulators(self.accumulators);
    }
}
