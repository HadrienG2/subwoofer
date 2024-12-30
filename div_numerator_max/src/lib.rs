use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
    operations::{self, Benchmark, Operation},
};
use rand::Rng;
use target_features::Architecture;

/// DIV with a possibly subnormal numerator, followed by MAX
#[derive(Clone, Copy)]
pub struct DivNumeratorMax;
//
impl<T: FloatLike> Operation<T> for DivNumeratorMax {
    const NAME: &str = "div_numerator_max";

    // One register for the 0.5 lower bound + a temporary on all CPU ISAs where
    // DIV cannot emit its output to the register that holds the denominator
    // TODO: Make sure only x86 got this silly idea
    fn aux_registers_regop(_input_registers: usize) -> usize {
        1 + cfg!(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            not(target_feature = "avx")
        )) as usize
    }

    // Still need the lower bound, the question is, can we use a memory input
    // for the numerator of a division?
    const AUX_REGISTERS_MEMOP: usize = const {
        let target = target_features::CURRENT_TARGET;
        match target.architecture() {
            // As of AVX-512 at least, x86 will not allow that
            Architecture::X86 => 2,
            // TODO: Check for other architectures
            _ => 1 + (!HAS_MEMORY_OPERANDS) as usize,
        }
    };

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
            operations::hide_accumulators::<_, ILP, false>,
            inputs,
            move |acc, elem| {
                // If elem is subnormal, the max takes us back to normal range,
                // otherwise this is a truncated multiplicative random walk that
                // cannot go lower than 0.5. In that case we can only use half
                // of the available exponent range but that's enough.
                operations::hide_single_accumulator(elem / acc).fast_max(T::splat(0.5))
            },
        );
    }

    #[inline]
    fn consume_outputs(self) {
        operations::consume_accumulators(self.accumulators);
    }
}
