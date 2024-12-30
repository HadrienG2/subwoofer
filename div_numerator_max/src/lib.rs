use common::{
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
    operations::{self, Benchmark, Operation},
};
use rand::Rng;

/// DIV with a possibly subnormal numerator, followed by MAX
#[derive(Clone, Copy)]
pub struct DivNumeratorMax;
//
impl<T: FloatLike> Operation<T> for DivNumeratorMax {
    const NAME: &str = "div_numerator_max";

    // One register for the 0.5 lower bound + a temporary on all CPU ISAs where
    // DIV cannot emit its output to the register that holds the denominator
    fn aux_registers_regop(_input_registers: usize) -> usize {
        1 + cfg!(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            not(target_feature = "avx")
        )) as usize
    }

    // I do not currently know of any architecture where we can use a memory
    // operand for a division's numerator, so we always need a temporary
    const AUX_REGISTERS_MEMOP: usize = 2;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        DivNumeratorMaxBenchmark {
            accumulators: [Default::default(); ILP],
        }
    }
}

/// [`Benchmark`] of [`MulMax`]
#[derive(Clone, Copy)]
struct DivNumeratorMaxBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for DivNumeratorMaxBenchmark<T, ILP> {
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
                // If elem is subnormal, the max takes us back to normal range,
                // otherwise this is a truncated multiplicative random walk that
                // cannot go lower than 0.5. In that case we can only use half of
                // the available exponent range but that's still plenty enough.
                (elem / acc).fast_max(T::splat(0.5))
            },
        );
    }

    #[inline]
    fn consume_outputs(self) {
        operations::consume_accumulators(self.accumulators);
    }
}
