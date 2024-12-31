use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
    operations::{self, Benchmark, Operation},
};
use rand::Rng;

/// Hardware maximum instruction with one data input.
///
/// When benchmarking dependency chains with a well-controlled share of
/// subnormal inputs, any operation that may produce subnormal or infinite
/// output must have its output be taken back to normal range with a cheap
/// hardware instruction before the the next iteration. We use hardware MIN/MAX
/// instructions (aka fast_min/max) for this purpose.
///
/// We therefore need to measure the overhead of MIN/MAX in isolation, so that
/// it can be subtracted from the overhed of another operation followed by
/// MIN/MAX to get the overhead of that operation in isolation. This benchmark
/// is enough for this, assuming hardware MIN and MAX have the same performance,
/// which is true of all currently known hardware.
#[derive(Clone, Copy)]
pub struct Max;
//
impl<T: FloatLike> Operation<T> for Max {
    const NAME: &str = "max";

    fn aux_registers_regop(_input_registers: usize) -> usize {
        0
    }

    // Inputs are directly reduced into the accumulator, can use memory operands
    const AUX_REGISTERS_MEMOP: usize = (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T> {
        MaxBenchmark {
            accumulators: [Default::default(); ILP],
        }
    }
}

/// [`Benchmark`] of [`Max`]
#[derive(Clone, Copy)]
struct MaxBenchmark<T: FloatLike, const ILP: usize> {
    accumulators: [T; ILP],
}
//
impl<T: FloatLike, const ILP: usize> Benchmark for MaxBenchmark<T, ILP> {
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
        // its knowledge that inputs are being reused.
        operations::integrate_full::<_, _, ILP, false>(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, true>,
            inputs,
            |acc, elem| acc.fast_max(elem),
        );
    }

    #[inline]
    fn consume_outputs(self) {
        operations::consume_accumulators(self.accumulators);
    }
}
