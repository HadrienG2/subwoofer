use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{generators::max::generate_max_inputs, Inputs, InputsMut},
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

    fn make_benchmark<Storage: InputsMut, const ILP: usize>(
        input_storage: Storage,
    ) -> impl Benchmark<Float = Storage::Element> {
        MaxBenchmark::<_, ILP> {
            input_storage,
            num_subnormals: None,
        }
    }
}

/// [`Benchmark`] of [`Max`]
struct MaxBenchmark<Storage: InputsMut, const ILP: usize> {
    input_storage: Storage,
    num_subnormals: Option<usize>,
}
//
impl<Storage: InputsMut, const ILP: usize> Benchmark for MaxBenchmark<Storage, ILP> {
    type Float = Storage::Element;

    fn num_operations(&self) -> usize {
        operations::accumulated_len(&self.input_storage, ILP)
    }

    fn setup_inputs(&mut self, num_subnormals: usize) {
        assert!(num_subnormals <= self.input_storage.as_ref().len());
        self.num_subnormals = Some(num_subnormals);
    }

    #[inline]
    fn start_run(&mut self, rng: &mut impl Rng) -> Self::Run<'_> {
        generate_max_inputs(
            self.input_storage.as_mut(),
            rng,
            self.num_subnormals
                .expect("setup_inputs should have been called first"),
        );
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

    fn inputs(&self) -> &[Self::Float] {
        self.inputs.as_ref()
    }

    #[inline]
    fn integrate_inputs(&mut self) {
        // No need to hide inputs for this benchmark, the compiler can't exploit
        // its knowledge that inputs are being reused.
        operations::integrate::<_, _, ILP, false>(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, true>,
            &mut self.inputs,
            // MAX is unaffected by the order of magnitude of inputs, so this
            // benchmark behaves homogeneously no matter what the order of
            // magnitude of its normal inputs is.
            |acc, elem| acc.fast_max(elem),
        );
    }

    #[inline]
    fn accumulators(&self) -> &[I::Element] {
        &self.accumulators
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::operations::test_utils::NeedsNarrowAcc;
    common::test_scalar_operation!(Max, NeedsNarrowAcc::Never);
}
