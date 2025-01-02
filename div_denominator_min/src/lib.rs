use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
    operations::{self, Benchmark, Operation},
};
use rand::Rng;

/// DIV with a possibly subnormal denominator, followed by MIN
#[derive(Clone, Copy)]
pub struct DivDenominatorMin;
//
impl<T: FloatLike> Operation<T> for DivDenominatorMin {
    const NAME: &str = "div_denominator_min";

    // One register for the upper bound
    fn aux_registers_regop(_input_registers: usize) -> usize {
        1
    }

    // Inputs are directly reduced into the accumulator, can use memory operands
    const AUX_REGISTERS_MEMOP: usize = 1 + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        DivDenominatorMinBenchmark {
            accumulators: [Default::default(); ILP],
        }
    }
}

/// [`Benchmark`] of [`MulMax`]
#[derive(Clone, Copy)]
struct DivDenominatorMinBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for DivDenominatorMinBenchmark<T, ILP> {
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
        // No need to hide inputs for this benchmark, the compiler can't exploit
        // its knowledge that inputs are being reused
        let upper_bound = upper_bound::<T>();
        operations::integrate_full::<_, _, ILP, false>(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, false>,
            inputs,
            move |acc, elem| {
                // - If elem is subnormal, the ratio is likely to be infinite
                //   but the MIN will get the accumulator back to normal range
                // - If elem is normal, this is effectively a multiplicative
                //   random walk of step [0.5; 2[ with an upper bound
                //   * It is a multiplicative random walk because the
                //     distribution of input values ensures that the
                //     distribution of 1/elem is the same as that of elem, and
                //     also (as a result) that repeatedly multiplying by elem
                //     results in a multiplicative random walk.
                //   * The random walk has a low chance of going below
                //     T::MIN_POSITIVE because input values are distributed in
                //     such a way that there is an equal chance of having elem
                //     and 1/elem in the input data stream, preventing long-term
                //     growth or shrinkage.
                operations::hide_single_accumulator(acc / elem).fast_min(upper_bound)
            },
        );
    }

    #[inline]
    fn consume_outputs(self) {
        operations::consume_accumulators(self.accumulators);
    }
}

/// Upper bound that we impose on accumulator values
///
/// Dividing by a subnormal number can produce infinite results, so we use a MIN
/// to get back to normal range. The upper bound should also allow us to avoid
/// transient infinities when dividing by normal numbers, so we want
/// `upper_bound / 0.5 < T::MAX`, i.e. `upper_bound > T::MAX / 2`.
///
/// On top of this fundamental limit, we add a /2 safety margin to protect
/// ourselves against rounding issues in e.g. the RNG in use.
fn upper_bound<T: FloatLike>() -> T {
    T::MAX / T::splat(4.0)
}
