use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
    operations::{self, Benchmark, Operation},
};
use rand::Rng;

/// MUL followed by MAX
#[derive(Clone, Copy)]
pub struct MulMax;
//
impl<T: FloatLike> Operation<T> for MulMax {
    const NAME: &str = "mul_max";

    // One register for the lower bound
    fn aux_registers_regop(_input_registers: usize) -> usize {
        1
    }

    // Inputs are directly reduced into the accumulator, can use memory operands
    const AUX_REGISTERS_MEMOP: usize = 1 + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        MulMaxBenchmark {
            accumulators: [Default::default(); ILP],
        }
    }
}

/// [`Benchmark`] of [`MulMax`]
#[derive(Clone, Copy)]
struct MulMaxBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for MulMaxBenchmark<T, ILP> {
    type Float = T;

    fn num_operations<Inputs: FloatSet>(inputs: &Inputs) -> usize {
        inputs.reused_len(ILP)
    }

    #[inline]
    fn begin_run(self, rng: impl Rng) -> Self {
        // We start close to the lower bound (at most 2x larger). This gives us
        // maximal headroom against hitting the T::MAX overflow limit.
        Self {
            accumulators: operations::normal_accumulators(rng)
                .map(|acc: T| (acc * T::splat(2.0)).sqrt() * lower_bound::<T>()),
        }
    }

    #[inline]
    fn integrate_inputs<Inputs>(&mut self, inputs: &mut Inputs)
    where
        Inputs: FloatSequence<Element = Self::Float>,
    {
        // No need to hide inputs for this benchmark, the compiler can't exploit
        // its knowledge that inputs are reused here
        let lower_bound = lower_bound::<T>();
        operations::integrate_full::<_, _, ILP, false>(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, false>,
            inputs,
            move |acc, elem| {
                // - If elem is subnormal, the MAX makes the output normal again
                // - If elem is normal, this is a multiplicative random walk of
                //   step [0.5; 2[ with a lower bound
                //   * This random walk is unlikely to go above T::MAX and
                //     overflow because input values are distributed in such a
                //     way that there is an equal chance of finding elem and
                //     1/elem in the input data stream, preventing long-term
                //     growth or shrinkage.
                operations::hide_single_accumulator(acc * elem).fast_max(lower_bound)
            },
        );
    }

    #[inline]
    fn consume_outputs(self) {
        operations::consume_accumulators(self.accumulators);
    }
}

/// Lower bound that we impose on accumulator values
///
/// Multiplying the accumulator by a subnormal number can produce a subnormal
/// result, so we use a MAX to get back to normal range. The lower bound that
/// this MAX imposes should also allow us to avoid subnormals when multiplying
/// by normal numbers, so we want `lower_bound * 0.5 > T::MIN_POSITIVE`, i.e.
/// `lower_bound > 2 * T::MIN_POSITIVE`.
///
/// On top of this fundamental limit, we add an x2 safety margin to protect
/// ourselves against rounding issues in e.g. the input data generator.
fn lower_bound<T: FloatLike>() -> T {
    T::splat(4.0) * T::MIN_POSITIVE
}
