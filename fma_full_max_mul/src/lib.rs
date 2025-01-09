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

    // One register for lower bound, one for the shrinking coefficient
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
            shrink: Default::default(),
        }
    }
}

/// [`Benchmark`] of [`FmaFullMaxMul`]
#[derive(Clone, Copy)]
struct FmaFullMaxMulBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
    shrink: T,
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for FmaFullMaxMulBenchmark<T, ILP> {
    type Float = T;

    fn num_operations<Inputs: FloatSet>(inputs: &Inputs) -> usize {
        inputs.reused_len(ILP) / 2
    }

    #[inline]
    fn begin_run(self, mut rng: impl Rng) -> Self {
        // Ensure that shrink is in the [1/8; 1/4[ range
        let normal_sampler = T::normal_sampler();
        let shrink = (normal_sampler(&mut rng) / T::splat(32.0)).sqrt();
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
        let lower_bound_preshrink = lower_bound_preshrink::<T>();
        let shrink = self.shrink;
        let iter = move |acc: T, factor, addend| {
            // - At the lower limit, if factor and addend are both subnormal,
            //   the FMA produces a subnormal result. This is then taken back to
            //   normal range by the MAX, and finally `shrink` reduces the
            //   result by a factor of 4 to 8.
            // - At the upper limit, factor and addend are both close to 2, so
            //   the FMA produces its maximal result `2*(acc + 1)`. The MAX does
            //   nothing to this normal value, then it is shrunk by a factor of
            //   1/8 to 1/4. In the upper limit scenario where `shrink = 1/4`,
            //   the result is therefore (acc / 2) + 1/4. Recursing this
            //   expression indefinitely would converge to the limit `sum(1/2^k)
            //   / 4`, which is 2/4 aka 1/2.
            // - For smaller factors and addends, we will converge to a smaller
            //   limit, which is enforced by MAX to be a normal number.
            operations::hide_single_accumulator(acc.mul_add(factor, addend))
                .fast_max(lower_bound_preshrink)
                * shrink
        };
        let inputs_slice = inputs.as_ref();
        let (factor_inputs, addend_inputs) = inputs_slice.split_at(inputs_slice.len() / 2);
        if Inputs::IS_REUSED {
            assert_eq!(factor_inputs.len(), addend_inputs.len());
            for (&factor, &addend) in factor_inputs.iter().zip(addend_inputs) {
                for acc in self.accumulators.iter_mut() {
                    *acc = iter(*acc, factor, addend);
                }
                operations::hide_accumulators::<_, ILP, true>(&mut self.accumulators);
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
                operations::hide_accumulators::<_, ILP, false>(&mut self.accumulators);
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
    fn accumulators(&self) -> &[T] {
        &self.accumulators
    }
}

/// Lower bound that we impose on accumulator values before shrinking
///
/// If the multiplier and the addend are both subnormal, then the output of the
/// FMA may be subnormal, so we use a MAX to get back to normal range. The lower
/// bound that this MAX imposes should have the following properties:
///
/// - The product of the lower bound by `shrink` should be a normal number.
///   Since `shrink >= 1/8`, this means that we need `lower_bound / 8 >
///   MIN_POSITIVE`, i.e. `lower_bound > 8 * MIN_POSITIVE`.
/// - To ensure that only subnormal inputs can cause the appearance of subnormal
///   temporaries inside of the FMA, the product of the smallest possible
///   accumulator `min_acc = lower_bound / 8` by the smallest possible normal
///   multiplier `min_factor = 0.5` should be a normal number. This means we
///   actually want `lower_bound > 16 * MIN_POSITIVE`.
///
/// As usual, we add an x2 safety margin on top of this fundamental limit to
/// protect ourselves against rounding issues in e.g. the RNG in use.
fn lower_bound_preshrink<T: FloatLike>() -> T {
    T::splat(32.0) * T::MIN_POSITIVE
}
