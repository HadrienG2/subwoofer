use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
    operations::{self, Benchmark, Operation},
};
use rand::Rng;

/// FMA with possibly subnormal multiplier, followed by averaging
#[derive(Clone, Copy)]
pub struct FmaMultiplierAverage;
//
impl<T: FloatLike> Operation<T> for FmaMultiplierAverage {
    const NAME: &str = "fma_multiplier_average";

    // One register for the averaging weight, another for the averaging target
    fn aux_registers_regop(_input_registers: usize) -> usize {
        2
    }

    // Inputs are directly reduced into the accumulator, can use memory operands
    const AUX_REGISTERS_MEMOP: usize = 2 + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        FmaMultiplierAverageBenchmark {
            accumulators: [Default::default(); ILP],
            target: Default::default(),
        }
    }
}

/// [`Benchmark`] of [`FmaMultiplierAverage`]
#[derive(Clone, Copy)]
struct FmaMultiplierAverageBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
    target: T,
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for FmaMultiplierAverageBenchmark<T, ILP> {
    type Float = T;

    fn num_operations<Inputs: FloatSet>(inputs: &Inputs) -> usize {
        inputs.reused_len(ILP)
    }

    #[inline]
    fn begin_run(self, mut rng: impl Rng) -> Self {
        let normal_sampler = T::normal_sampler();
        let target = normal_sampler(&mut rng);
        let accumulators = operations::normal_accumulators(rng);
        Self {
            accumulators,
            target,
        }
    }

    #[inline]
    fn integrate_inputs<Inputs>(&mut self, inputs: &mut Inputs)
    where
        Inputs: FloatSequence<Element = Self::Float>,
    {
        let halve_weight = T::splat(0.5);
        let target = self.target;
        operations::integrate_full(&mut self.accumulators, inputs, move |acc, elem| {
            (acc.mul_add(elem, halve_weight) + target) * halve_weight
        });
    }

    #[inline]
    fn consume_outputs(self) {
        operations::consume_accumulators(self.accumulators);
    }
}
