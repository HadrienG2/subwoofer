use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
    operations::{self, Benchmark, Operation},
};
use rand::Rng;

/// FMA with possibly subnormal inputs, followed by MAX/MUL to restore acc range
#[derive(Clone, Copy)]
pub struct FmaFullMaxMul;
//
impl<T: FloatLike> Operation<T> for FmaFullMaxMul {
    const NAME: &str = "fma_full_max_mul";

    // One register for lower bound 2.0, one for magnitude reduction coeff 0.25
    fn aux_registers_regop(_input_registers: usize) -> usize {
        2
    }

    // Accumulator is not reused after FMA, so can use one memory operand. But
    // no known hardware supports two memory operands for FMA, so we still need
    // to reserve one register for loading the other input.
    const AUX_REGISTERS_MEMOP: usize = 3 + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        FmaFullMaxMulBenchmark {
            accumulators: [Default::default(); ILP],
        }
    }
}

/// [`Benchmark`] of [`FmaFullMaxMul`]
#[derive(Clone, Copy)]
struct FmaFullMaxMulBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for FmaFullMaxMulBenchmark<T, ILP> {
    type Float = T;

    fn num_operations<Inputs: FloatSet>(inputs: &Inputs) -> usize {
        inputs.reused_len(ILP) / 2
    }

    #[inline]
    fn begin_run(self, rng: impl Rng) -> Self {
        Self {
            // Accumulators are initially in range [0.5; 1[
            accumulators: operations::normal_accumulators(rng)
                .map(|acc: T| (acc * T::splat(0.5)).sqrt()),
        }
    }

    #[inline]
    fn integrate_inputs<Inputs>(&mut self, inputs: &mut Inputs)
    where
        Inputs: FloatSequence<Element = Self::Float>,
    {
        let iter = move |acc: T, factor, addend| {
            // If we assume that acc is initially in range [0.5; 1[, then it
            // will stay there across benchmark iterations because...
            //
            // - factor and addend are in range [0; 2[, therefore if acc is
            //   initially in range [0.5; 1.0[, acc.mul_add(factor, addend) is
            //   in range [0; 1*2+2[ = [0; 4[.
            // - By applying the maximum, we get back to normal range [2; 4[
            // - By dividing by 4, we get back to initial acc range [0.5; 1[
            acc.mul_add(factor, addend).fast_max(T::splat(2.0)) * T::splat(0.25)
        };
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
