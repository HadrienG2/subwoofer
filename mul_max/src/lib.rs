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
        Self {
            accumulators: operations::narrow_accumulators(rng),
        }
    }

    #[inline]
    fn integrate_inputs<Inputs>(&mut self, inputs: &mut Inputs)
    where
        Inputs: FloatSequence<Element = T>,
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
    fn accumulators(&self) -> &[T] {
        &self.accumulators
    }
}

/// Lower bound that we impose on accumulator values
///
/// Multiplying the accumulator by a subnormal number can produce a subnormal
/// result, so we use a MAX to get back to normal range.
///
/// The lower bound that this MAX imposes should additionally ensure that the
/// product of a subnormal number by the accumulator has a good chance of being
/// a subnormal number other than 0, because some hardware has no issue with
/// multiplications by a subnormal that lead to a normal or zero result.
/// However, there is a tradeoff here:
///
/// - When the accumulator is exactly 1.0, the output is a subnormal number if
///   and only if the input is one.
/// - As the accumulator grows above 1.0, multiplying by some subnormal inputs
///   leads to a normal output. This becomes always true once the accumulator
///   grows above 2^T::MANTISSA_DIGITS.
/// - As the accumulator shrinks below 1.0, multiplying by some subnormal inputs
///   leads a zero output. This becomes always true once the accumulator shrinks
///   below 2^-T::MANTISSA_DIGITS.
/// - All other things being equal, it would be good to have a lower bound that
///   constrains the accumulator as little as possible, so that the accumulator
///   is allowed to explore more of the available range of values and has more
///   room to grow before overflowing.
///
/// A lower bound that is at the multiplicative half-way point between 1.0 and
/// 2^-T::MANTISSA_DIGITS seems like it strikes a good balance between these
/// various concerns.
fn lower_bound<T: FloatLike>() -> T {
    T::splat(2.0f32.powi(-(T::MANTISSA_DIGITS as i32) / 2))
}
