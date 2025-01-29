use common::{
    arch::{HAS_HARDWARE_NEGATED_FMA, HAS_MEMORY_OPERANDS},
    floats::{self, FloatLike},
    inputs::{generators::add::generate_add_inputs, Inputs, InputsMut},
    operations::{self, Benchmark, BenchmarkRun, Operation},
};
use rand::Rng;

/// FMA with possibly subnormal multiplier
#[derive(Clone, Copy)]
pub struct FmaMultiplier;
//
impl Operation for FmaMultiplier {
    const NAME: &str = "fma_multiplier";

    // One register for the constant multiplier, and another register for a
    // negated version of that multiplier if the target CPU does not have an
    // FNMA instruction
    fn aux_registers_regop(_input_registers: usize) -> usize {
        1 + (!HAS_HARDWARE_NEGATED_FMA) as usize
    }

    // Acccumulator is not reused after FMA, so we can use memory operands if
    // available to reduce register pressure
    const AUX_REGISTERS_MEMOP: usize =
        1 + (!HAS_HARDWARE_NEGATED_FMA) as usize + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<Storage: InputsMut, const ILP: usize>(
        input_storage: Storage,
    ) -> impl Benchmark<Float = Storage::Element> {
        FmaMultiplierBenchmark::<_, ILP> {
            input_storage,
            num_subnormals: None,
        }
    }
}

/// [`Benchmark`] of [`FmaMultiplier`]
struct FmaMultiplierBenchmark<Storage: InputsMut, const ILP: usize> {
    input_storage: Storage,
    num_subnormals: Option<usize>,
}
//
impl<Storage: InputsMut, const ILP: usize> Benchmark for FmaMultiplierBenchmark<Storage, ILP> {
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
        generate_add_inputs::<_, ILP>(
            &mut self.input_storage,
            rng,
            self.num_subnormals
                .expect("Should have called setup_inputs first"),
        );
        let narrow = floats::narrow_sampler();
        FmaMultiplierRun {
            inputs: self.input_storage.freeze(),
            accumulators: operations::narrow_accumulators(rng),
            multiplier: narrow(rng),
        }
    }

    type Run<'run>
        = FmaMultiplierRun<Storage::Frozen<'run>, ILP>
    where
        Self: 'run;
}

/// [`BenchmarkRun`] of [`FmaMultiplier`]
struct FmaMultiplierRun<I: Inputs, const ILP: usize> {
    inputs: I,
    accumulators: [I::Element; ILP],
    multiplier: I::Element,
}
//
impl<Storage: Inputs, const ILP: usize> BenchmarkRun for FmaMultiplierRun<Storage, ILP> {
    type Float = Storage::Element;

    fn inputs(&self) -> &[Self::Float] {
        self.inputs.as_ref()
    }

    #[inline]
    fn integrate_inputs(&mut self) {
        // Overall, this is just the add benchmark with a step size that is at
        // most 2x smaller/larger, which fundamentally doesn't change much
        let multiplier = self.multiplier;
        operations::integrate::<_, _, ILP, false>(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, false>,
            &mut self.inputs,
            // - Each normal input `elem` is followed by the opposite input
            //   `-elem`, taking the accumulator back to its nominal magnitude
            //   range [1/2; 2[ since the multiplier is always the same.
            // - Subnormal inputs have no effect on the accumulator
            move |acc, elem| elem.mul_add(multiplier, acc),
        );
    }

    #[inline]
    fn accumulators(&self) -> &[Storage::Element] {
        &self.accumulators
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::operations::test_utils::NeedsNarrowAcc;
    common::test_scalar_operation!(FmaMultiplier, NeedsNarrowAcc::Always, 20.0 * f32::EPSILON);
}
