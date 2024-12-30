use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
    operations::{self, Benchmark, Operation},
};
use rand::Rng;

/// FMA with possibly subnormal multiplier, followed by MIN
#[derive(Clone, Copy)]
pub struct FmaMultiplierMin;
//
impl<T: FloatLike> Operation<T> for FmaMultiplierMin {
    const NAME: &str = "fma_multiplier_min";

    // One register for the normal addend, one for 2.0 upper bound
    fn aux_registers_regop(_input_registers: usize) -> usize {
        2
    }

    // Acccumulator is not reused after FMA, so we can use memory operands if
    // available to reduce register pressure
    const AUX_REGISTERS_MEMOP: usize = 2 + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        FmaMultiplierMinBenchmark {
            accumulators: [Default::default(); ILP],
            normal: Default::default(),
        }
    }
}

/// [`Benchmark`] of [`FmaMultiplierMin`]
#[derive(Clone, Copy)]
struct FmaMultiplierMinBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
    normal: T,
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for FmaMultiplierMinBenchmark<T, ILP> {
    type Float = T;

    fn num_operations<Inputs: FloatSet>(inputs: &Inputs) -> usize {
        inputs.reused_len(ILP)
    }

    #[inline]
    fn begin_run(self, mut rng: impl Rng) -> Self {
        let normal_sampler = T::normal_sampler();
        let normal = normal_sampler(&mut rng);
        let accumulators = operations::normal_accumulators(rng);
        Self {
            accumulators,
            normal,
        }
    }

    #[inline]
    fn integrate_inputs<Inputs>(&mut self, inputs: &mut Inputs)
    where
        Inputs: FloatSequence<Element = Self::Float>,
    {
        let normal = self.normal;
        // No need to hide inputs for this benchmark, compiler can't exploit
        // knowledge of input reuse here
        operations::integrate_full::<_, _, ILP, false>(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, false>,
            inputs,
            move |acc, elem| {
                // Because acc and normal are initially in range [0.5; 2[, they will
                // stay forever in this range. Here is why:
                //
                // - If elem is subnormal, this is the MIN of normal and 2,
                //   which is just normal since normal can't be greater than 2.
                // - If elem is normal, acc * elem is in range [0.5*acc, 2*acc[.
                //   When acc is in range [0.5; 2[, this product is in [0.25; 4[
                // - Because normal is in range [0.5; 2[, it follows from the
                //   previous computation that acc*elem+normal is in [0.75; 6[.
                // - After minimum with acc, which is smaller than 2, we are
                //   thus in range [0.75; 2[, and this is a subset of initial
                //   range [0.5; 2[, which is what we wanted
                operations::hide_single_accumulator(acc.mul_add(elem, normal))
                    .fast_min(T::splat(2.0))
            },
        );
    }

    #[inline]
    fn consume_outputs(self) {
        operations::consume_accumulators(self.accumulators);
    }
}
