use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
    operations::{self, Benchmark, Operation},
};
use rand::Rng;

/// FMA with possibly subnormal addend, followed by MIN
#[derive(Clone, Copy)]
pub struct FmaAddendMin;
//
impl<T: FloatLike> Operation<T> for FmaAddendMin {
    const NAME: &str = "fma_addend_min";

    // One register for the growth factor, one for the 2.0 upper bound
    fn aux_registers_regop(_input_registers: usize) -> usize {
        2
    }

    // acc is not reused after FMA, so memory operands can be used. Without
    // memory operands, need to load elem into another register before FMA.
    const AUX_REGISTERS_MEMOP: usize = 2 + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        FmaAddendMinBenchmark {
            accumulators: [Default::default(); ILP],
            growth: Default::default(),
        }
    }
}

/// [`Benchmark`] of [`FmaAddendMin`]
#[derive(Clone, Copy)]
struct FmaAddendMinBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
    growth: T,
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for FmaAddendMinBenchmark<T, ILP> {
    type Float = T;

    fn num_operations<Inputs: FloatSet>(inputs: &Inputs) -> usize {
        inputs.reused_len(ILP)
    }

    #[inline]
    fn begin_run(self, mut rng: impl Rng) -> Self {
        let normal_sampler = T::normal_sampler();
        // Ensure that growth is in the [1; 2[ range
        let growth = (normal_sampler(&mut rng) * T::splat(2.0)).sqrt();
        let accumulators = operations::normal_accumulators(rng);
        Self {
            accumulators,
            growth,
        }
    }

    #[inline]
    fn integrate_inputs<Inputs>(&mut self, inputs: &mut Inputs)
    where
        Inputs: FloatSequence<Element = Self::Float>,
    {
        let growth = self.growth;
        // No need to hide inputs for this benchmark, compiler can't exploit
        // knowledge of input reuse here
        operations::integrate_full::<_, _, ILP, false>(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, false>,
            inputs,
            move |acc, elem| {
                // Given that acc > 0 and growth >= 1, acc * growth >= acc.
                // Because elem > 0, if follows that acc*growth+elem > acc. So
                // acc cannot shrink, and its growth is also bounded by MIN.
                // Hence it will forever remain in its initial [0.5; 2[ range.
                operations::hide_single_accumulator(acc.mul_add(growth, elem))
                    .fast_min(T::splat(2.0))
            },
        );
    }

    #[inline]
    fn consume_outputs(self) {
        operations::consume_accumulators(self.accumulators);
    }
}
