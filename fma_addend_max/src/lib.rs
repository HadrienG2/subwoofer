use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
    operations::{self, Benchmark, Operation},
};
use rand::Rng;

/// FMA with possibly subnormal addend, followed by MAX
#[derive(Clone, Copy)]
pub struct FmaAddendMax;
//
impl<T: FloatLike> Operation<T> for FmaAddendMax {
    const NAME: &str = "fma_addend_max";

    // One register for the shrinking factor, one for the lower bound
    fn aux_registers_regop(_input_registers: usize) -> usize {
        2
    }

    // acc is not reused after FMA, so memory operands can be used. Without
    // memory operands, need to load elem into another register before FMA.
    const AUX_REGISTERS_MEMOP: usize = 2 + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        FmaAddendMaxBenchmark {
            accumulators: [Default::default(); ILP],
            shrink: Default::default(),
        }
    }
}

/// [`Benchmark`] of [`FmaAddendMax`]
#[derive(Clone, Copy)]
struct FmaAddendMaxBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
    shrink: T,
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for FmaAddendMaxBenchmark<T, ILP> {
    type Float = T;

    fn num_operations<Inputs: FloatSet>(inputs: &Inputs) -> usize {
        inputs.reused_len(ILP)
    }

    #[inline]
    fn begin_run(self, mut rng: impl Rng) -> Self {
        // Ensure that shrink is in the [1/4; 1/2[ range
        let normal_sampler = T::normal_sampler();
        let shrink = (normal_sampler(&mut rng) / T::splat(8.0)).sqrt();
        let accumulators = operations::narrow_accumulators(rng);
        Self {
            accumulators,
            shrink,
        }
    }

    #[inline]
    fn integrate_inputs<Inputs>(&mut self, inputs: &mut Inputs)
    where
        Inputs: FloatSequence<Element = T>,
    {
        // No need to hide inputs for this benchmark, compiler can't exploit
        // knowledge of input reuse here
        let shrink = self.shrink;
        let lower_bound = lower_bound::<T>();
        operations::integrate_full::<_, _, ILP, false>(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, false>,
            inputs,
            move |acc, elem| {
                // We enforce `shrink <= 0.5`, so...
                //
                // - In the lower limit scenario where all inputs are subnormal
                //   numbers, acc decays by virtue of being repeatedly
                //   multiplied by `shrink < 1`, until it hits the minimum
                //   imposed by `lower_bound` and stays there
                // - In the upper limit scenario where `shrink` is 0.5 and all
                //   inputs are close to 2, acc converges to `2 * sum(1/2^k)`,
                //   which is `2 * 1/2` aka 1.
                // - For smaller values of `shrink` and smaller input data, the
                //   accumulator will converge to a smaller limit, which the
                //   lower bound forces to be a normal number.
                operations::hide_single_accumulator(acc.mul_add(shrink, elem)).fast_max(lower_bound)
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
/// We want to ensure that the `acc * shrink` product is a normal number. Since
/// `shrink` is between 1/4 and 1/2, this means that `acc` must not be allowed
/// to become smaller than `4 * T::MIN_POSITIVE`.
///
/// On top of this fundamental limit, we add a 2x safety margin to protect
/// ourselves against rounding issues in e.g. the input data generator.
fn lower_bound<T: FloatLike>() -> T {
    T::splat(8.0) * T::MIN_POSITIVE
}
