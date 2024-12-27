use common::{
    inputs::{FloatSequence, FloatSet},
    process::{self, Benchmark, Operation},
    types::FloatLike,
};
use rand::Rng;

/// Square root of positive numbers, followed by ADD/SUB
///
/// Square roots of negative numbers may or may not be emulated in software.
/// They are thus not a good candidate for CPU microbenchmarking.
#[derive(Clone, Copy)]
pub struct SqrtPositiveAddSub;
//
impl<T: FloatLike> Operation<T> for SqrtPositiveAddSub {
    const NAME: &str = "sqrt_positive_addsub";

    // Square root goes to a temporary before use, even with memory operands
    const AUX_REGISTERS_REGOP: usize = 1;
    //
    const AUX_REGISTERS_MEMOP: usize = 1;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        SqrtPositiveAddSubBenchmark {
            accumulators: [Default::default(); ILP],
        }
    }
}

/// [`Benchmark`] of [`SqrtPositiveAddSub`]
#[derive(Clone, Copy)]
struct SqrtPositiveAddSubBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for SqrtPositiveAddSubBenchmark<T, ILP> {
    type Float = T;

    fn num_operations<Inputs: FloatSet>(inputs: &Inputs) -> usize {
        inputs.reused_len(ILP)
    }

    fn begin_run(&mut self, rng: impl Rng) {
        // This is just a random additive walk of ~unity or subnormal step, so given a
        // high enough starting point, an initially normal accumulator should stay in
        // the normal range forever.
        self.accumulators = process::additive_accumulators(rng);
    }

    #[inline]
    fn integrate_inputs<Inputs>(mut self, inputs: Inputs) -> (Self, Inputs)
    where
        Inputs: FloatSequence<Element = Self::Float>,
    {
        let low_iter = |acc, elem: T| acc + elem.sqrt();
        let high_iter = |acc, elem: T| acc - elem.sqrt();
        let (next_accs, next_inputs) = if Inputs::IS_REUSED {
            // Need to hide reused register inputs, so that the compiler
            // doesn't abusively factor out the redundant square root
            // computations and reuse their result for all accumulators (in
            // fact it would even be allowed to reuse them for the entire
            // outer iters loop in run_benchmark).
            process::integrate_halves::<_, _, ILP, true>(
                self.accumulators,
                inputs,
                low_iter,
                high_iter,
            )
        } else {
            // Memory inputs do not need to be hidden because each
            // accumulator gets its own input substream (preventing square
            // root reuse during the inner loop over accumulators) and
            // current LLVM is not crazy enough to precompute square roots
            // for a whole arbitrarily large dynamically-sized batch of
            // input data.
            assert!(Inputs::NUM_REGISTER_INPUTS.is_none());
            process::integrate_halves::<_, _, ILP, false>(
                self.accumulators,
                inputs,
                low_iter,
                high_iter,
            )
        };
        self.accumulators = next_accs;
        (self, next_inputs)
    }

    fn consume_outputs(self) {
        process::consume_accumulators(self.accumulators);
    }
}
