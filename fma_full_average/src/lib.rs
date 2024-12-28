use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
    operations::{self, Benchmark, Operation},
};
use rand::Rng;

/// FMA with possibly subnormal inputs, followed by averaging
#[derive(Clone, Copy)]
pub struct FmaFullAverage;
//
impl<T: FloatLike> Operation<T> for FmaFullAverage {
    const NAME: &str = "fma_full_average";

    // One register for the averaging weight, another for the averaging target
    const AUX_REGISTERS_REGOP: usize = 2;

    // Inputs are directly reduced into the accumulator, so we can use a memory
    // operand, but even on architectures with memory operands FMA does not
    // support two memory operands so we still need at least one input register
    const AUX_REGISTERS_MEMOP: usize = 3 + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        FmaFullAverageBenchmark {
            accumulators: [Default::default(); ILP],
            target: Default::default(),
        }
    }
}

/// [`Benchmark`] of [`FmaFullAverage`]
#[derive(Clone, Copy)]
struct FmaFullAverageBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
    target: T,
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for FmaFullAverageBenchmark<T, ILP> {
    type Float = T;

    fn num_operations<Inputs: FloatSet>(inputs: &Inputs) -> usize {
        inputs.reused_len(ILP) / 2
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
        let target = self.target;
        let iter =
            move |acc: T, factor, addend| (acc.mul_add(factor, addend) + target) * T::splat(0.5);
        let inputs_slice = inputs.as_ref();
        let (factor_inputs, addend_inputs) = inputs_slice.split_at(inputs_slice.len() / 2);
        if Inputs::IS_REUSED {
            assert_eq!(factor_inputs.len(), addend_inputs.len());
            for (&factor, &addend) in factor_inputs.iter().zip(addend_inputs) {
                for acc in self.accumulators.iter_mut() {
                    *acc = iter(*acc, factor, addend);
                }
                operations::hide_accumulators(&mut self.accumulators);
            }
        } else {
            let factor_chunks = factor_inputs.chunks_exact(ILP);
            let addend_chunks = addend_inputs.chunks_exact(ILP);
            let factor_remainder = factor_chunks.remainder();
            let addend_remainder = addend_chunks.remainder();
            for (factor_chunk, addend_chunk) in factor_chunks.zip(addend_chunks) {
                for ((&factor, &addend), acc) in factor_chunk
                    .iter()
                    .zip(addend_chunk)
                    .zip(self.accumulators.iter_mut())
                {
                    *acc = iter(*acc, factor, addend);
                }
                operations::hide_accumulators(&mut self.accumulators);
            }
            for ((&factor, &addend), acc) in factor_remainder
                .iter()
                .zip(addend_remainder)
                .zip(self.accumulators.iter_mut())
            {
                *acc = iter(*acc, factor, addend);
            }
        }
    }

    #[inline]
    fn consume_outputs(self) {
        operations::consume_accumulators(self.accumulators);
    }
}
