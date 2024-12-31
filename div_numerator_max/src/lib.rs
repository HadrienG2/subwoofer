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
            accumulators: operations::normal_accumulators(rng),
        }
    }

    #[inline]
    fn integrate_inputs<Inputs>(&mut self, inputs: &mut Inputs)
    where
        Inputs: FloatSequence<Element = Self::Float>,
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
    fn consume_outputs(self) {
        operations::consume_accumulators(self.accumulators);
    }
}

/// Lower bound that we impose on accumulator values
///
/// Dividing a subnormal number by our accumulator can produce a subnormal
/// result, so we use a MAX to get back to normal range. The lower bound is
/// chosen to ensure that...
///
/// - The resulting accumulator is a normal number, above `T::MIN_POSITIVE`.
/// - `elem / acc` cannot overflow when acc is at this lower bound. Since `elem
///   <= 2`, this means that we want `2 / lower_bound < T::MAX` i.e.
///   `lower_bound > 2 / T::MAX`.
/// - At this upper limit of the accumulator range `max_acc = 2 / lower_bound`,
///   the `elem / acc` ratio should not underflow below `T::MIN_POSITIVE` when
///   elem is normal. Since `elem >= 0.5` in that case, this requires `0.5 /
///   max_acc > T::MIN_POSITIVE` i.e. `lower_bound > 4 * T::MIN_POSITIVE`.
///
/// We add a 2x safety margin on top of these limits to protect ourselves
/// against rounding issues in e.g. the RNG we use.
fn lower_bound<T: FloatLike>() -> T {
    (T::splat(8.0) * T::MIN_POSITIVE).fast_max(T::splat(4.0) / T::MAX)
}
