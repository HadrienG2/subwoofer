use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{self, Inputs, InputsMut},
    operations::{self, Benchmark, BenchmarkRun, Operation},
};
use rand::prelude::*;

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
impl Operation for Max {
    const NAME: &str = "max";

    fn aux_registers_regop(_input_registers: usize) -> usize {
        0
    }

    // Inputs are directly reduced into the accumulator, can use memory operands
    const AUX_REGISTERS_MEMOP: usize = (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>(input_storage: impl InputsMut) -> impl Benchmark {
        MaxBenchmark::<_, ILP> {
            input_storage,
            num_subnormals: None,
        }
    }
}

/// [`Benchmark`] of [`Max`]
#[derive(Clone, Copy)]
struct MaxBenchmark<Storage: InputsMut, const ILP: usize> {
    input_storage: Storage,
    num_subnormals: Option<usize>,
}
//
impl<Storage: InputsMut, const ILP: usize> Benchmark for MaxBenchmark<Storage, ILP> {
    fn num_operations(&self) -> usize {
        inputs::accumulated_len(&self.input_storage, ILP)
    }

    const SUBNORMAL_INPUT_GRANULARITY: usize = 1;

    fn setup_inputs(&mut self, rng: &mut impl Rng, num_subnormals: usize) {
        // Decide whether to generate inputs now or every run
        if operations::should_generate_every_run(&self.input_storage, num_subnormals, ILP, 1) {
            self.num_subnormals = Some(num_subnormals);
        } else {
            inputs::generate_mixture(self.input_storage.as_mut(), rng, num_subnormals);
            self.num_subnormals = None;
        }
    }

    #[inline]
    fn start_run(&mut self, rng: &mut impl Rng) -> Self::Run<'_> {
        // Generate inputs if needed, otherwise just shuffle them around
        if let Some(num_subnormals) = self.num_subnormals {
            inputs::generate_mixture(self.input_storage.as_mut(), rng, num_subnormals);
        } else {
            self.input_storage.as_mut().shuffle(rng);
        }

        // For this benchmark, the initial value of accumulators does not
        // matter, as long as it is normal.
        MaxRun {
            inputs: self.input_storage.freeze(),
            accumulators: operations::normal_accumulators(rng),
        }
    }

    type Run<'run>
        = MaxRun<Storage::Frozen<'run>, ILP>
    where
        Self: 'run;
}

/// [`BenchmarkRun`] of [`Max`]
struct MaxRun<I: Inputs, const ILP: usize> {
    inputs: I,
    accumulators: [I::Element; ILP],
}
//
impl<I: Inputs, const ILP: usize> BenchmarkRun for MaxRun<I, ILP> {
    type Float = I::Element;

    #[inline]
    fn integrate_inputs(&mut self) {
        // No need to hide inputs for this benchmark, the compiler can't exploit
        // its knowledge that inputs are being reused.
        operations::integrate_full::<_, _, ILP, false>(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, true>,
            &mut self.inputs,
            |acc, elem| acc.fast_max(elem),
        );
    }

    #[inline]
    fn accumulators(&self) -> &[I::Element] {
        &self.accumulators
    }
}
