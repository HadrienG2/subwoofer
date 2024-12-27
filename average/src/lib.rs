use common::{
    arch::HAS_MEMORY_OPERANDS,
    inputs::{FloatSequence, FloatSet},
    process::{self, Benchmark, Operation},
    types::FloatLike,
};
use rand::Rng;

/// Rolling average with some data inputs.
///
/// For multiplicative benchmarks, we're going to need to an extra averaging
/// operation, otherwise once we've multiplied by a subnormal we'll stay in
/// subnormal range forever.
///
/// It cannot be just an addition, because otherwise if we have no subnormal
/// input we get unbounded growth, which is also a problem.
///
/// This benchmark measures the overhead of averaging with an in-register input
/// in isolation, so that it can be subtracted from the overhead of X followed
/// by averaging (with due respect paid to the existence of superscalar
/// execution).
#[derive(Clone, Copy)]
pub struct Average;
//
impl<T: FloatLike> Operation<T> for Average {
    const NAME: &str = "average";

    // Need a register to hold the averaging weight 0.5
    const AUX_REGISTERS_REGOP: usize = 1;

    // Inputs are directly reduced into the accumulator, can use memory operands
    const AUX_REGISTERS_MEMOP: usize = 1 + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        AverageBenchmark {
            accumulators: [Default::default(); ILP],
        }
    }
}

/// [`Benchmark`] of [`Average`]
#[derive(Clone, Copy)]
struct AverageBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for AverageBenchmark<T, ILP> {
    type Float = T;

    fn num_operations<Inputs: FloatSet>(inputs: &Inputs) -> usize {
        inputs.reused_len(ILP)
    }

    fn begin_run(&mut self, rng: impl Rng) {
        self.accumulators = process::multiplicative_accumulators(rng);
    }

    #[inline]
    fn integrate_inputs<Inputs>(mut self, inputs: Inputs) -> (Self, Inputs)
    where
        Inputs: FloatSequence<Element = Self::Float>,
    {
        let (next_accs, next_inputs) =
            process::integrate_full(self.accumulators, inputs, |acc, elem| {
                (acc + elem) * T::splat(0.5)
            });
        self.accumulators = next_accs;
        (self, next_inputs)
    }

    fn consume_outputs(self) {
        process::consume_accumulators(self.accumulators);
    }
}
