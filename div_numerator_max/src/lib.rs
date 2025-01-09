use common::{
    arch::{ALLOWS_DIV_MEMORY_NUMERATOR, ALLOWS_DIV_OUTPUT_DENOMINATOR},
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

    // One register for the lower bound + a temporary on all CPU ISAs where DIV
    // cannot emit its output to the register that holds the denominator
    fn aux_registers_regop(_input_registers: usize) -> usize {
        1 + ALLOWS_DIV_OUTPUT_DENOMINATOR as usize
    }

    // Still need the lower bound register. But the question is, can we use a
    // memory input for the numerator of a division?
    const AUX_REGISTERS_MEMOP: usize = 1 + (!ALLOWS_DIV_MEMORY_NUMERATOR) as usize;

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
            accumulators: operations::narrow_accumulators(rng),
        }
    }

    #[inline]
    fn integrate_inputs<Inputs>(&mut self, inputs: &mut Inputs)
    where
        Inputs: FloatSequence<Element = T>,
    {
        // No need to hide inputs for this benchmark, the compiler can't exploit
        // its knowledge that inputs are being reused here
        let lower_bound = lower_bound::<T>();
        operations::integrate_full::<_, _, ILP, false>(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, false>,
            inputs,
            move |acc, elem| {
                // - If elem is subnormal, the MAX makes the output normal again
                // - If elem is normal, this is a weird kind of random wolk
                //   * ...but however weird, this walk is bounded on both sides
                //     because by imposing a minimal accumulator value, we also
                //     impose a lower bound on the denominator of the division,
                //     and thus an upper limit to the ratio that will become the
                //     next accumulator. See the docs of lower_bound below for a
                //     longer demonstration.
                operations::hide_single_accumulator(elem / acc).fast_max(lower_bound)
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
/// Dividing a subnormal number by our accumulator can produce a subnormal
/// result, so we use a MAX to get back to normal range.
///
/// The lower bound that this MAX imposes should additionally ensure that the
/// ratio of a subnormal number by the accumulator has a good chance of being a
/// subnormal number other than 0, because some hardware has no issue with
/// divisions that have a subnormal numerator but lead to a normal or zero
/// result. However, there is a tradeoff here:
///
/// - When the accumulator is exactly 1.0, the output is a subnormal number if
///   and only if the input is one.
/// - As the accumulator grows above 1.0, dividing some subnormal inputs by the
///   accumulator leads to a zero output. This becomes always true once the
///   accumulator grows above 2^T::MANTISSA_DIGITS.
/// - As the accumulator shrinks below 1.0, dividing some subnormal inputs by
///   the accumulator leads to a normal output. This becomes always true once
///   the accumulator shrinks below 2^-T::MANTISSA_DIGITS.
/// - When we give the accumulator a lower bound, we also give it an upper
///   bound: since elem cannot be higher than 2.0, given an initial accumulator
///   value `acc`, the next accumulator value `elem / acc` cannot be higher than
///   `2 / lower_bound`.
///
/// A lower bound that is at the multiplicative half-way point between 1.0 and
/// 2^-T::MANTISSA_DIGITS seems like it strikes a good balance between these
/// various concerns.
fn lower_bound<T: FloatLike>() -> T {
    T::splat(2.0f32.powi(-(T::MANTISSA_DIGITS as i32) / 2))
}
