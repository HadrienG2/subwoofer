use common::{
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
    operations::{self, Benchmark, Operation},
};
use rand::Rng;

/// Square root of positive numbers, followed by MAX
///
/// Square roots of negative numbers may or may not be emulated in software.
/// They are thus not a good candidate for CPU microbenchmarking.
#[derive(Clone, Copy)]
pub struct SqrtPositiveMax;
//
impl<T: FloatLike> Operation<T> for SqrtPositiveMax {
    const NAME: &str = "sqrt_positive_max";

    // A square root must go to a temporary before use...
    fn aux_registers_regop(input_registers: usize) -> usize {
        // ...and unfortunately, it looks like the input optimization barrier
        // that we apply to avoid sqrt-hoisting has the side-effect of
        // pessimizing this into a full doubling of register pressure.
        input_registers
    }

    // ...but architectures without memory operands can reuse the register that
    // was used to load the input in order to hold the square root.
    const AUX_REGISTERS_MEMOP: usize = 1;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        SqrtPositiveMaxBenchmark {
            accumulators: [Default::default(); ILP],
        }
    }
}

/// [`Benchmark`] of [`SqrtPositiveMax`]
#[derive(Clone, Copy)]
struct SqrtPositiveMaxBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for SqrtPositiveMaxBenchmark<T, ILP> {
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
        let iter = |acc: T, elem: T| acc.fast_max(elem.sqrt());
        if Inputs::IS_REUSED {
            // Need to hide reused register inputs, so that the compiler
            // doesn't abusively factor out the redundant square root
            // computations and reuse their result for all accumulators (in
            // fact it would even be allowed to reuse them for the entire
            // outer iters loop in run_benchmark).
            operations::integrate_full::<_, _, ILP, true>(&mut self.accumulators, inputs, iter)
        } else {
            // Memory inputs do not need to be hidden because each
            // accumulator gets its own input substream (preventing square
            // root reuse during the inner loop over accumulators) and
            // current LLVM is not crazy enough to precompute square roots
            // for a whole arbitrarily large dynamically-sized batch of
            // input data.
            assert!(Inputs::NUM_REGISTER_INPUTS.is_none());
            operations::integrate_full::<_, _, ILP, false>(&mut self.accumulators, inputs, iter)
        };
    }

    #[inline]
    fn consume_outputs(self) {
        operations::consume_accumulators(self.accumulators);
    }
}
