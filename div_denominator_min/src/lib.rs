use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{generators::muldiv::generate_muldiv_inputs, Inputs, InputsMut},
    operations::{self, Benchmark, BenchmarkRun, Operation},
};
use rand::Rng;

/// DIV with a possibly subnormal denominator, followed by MIN
#[derive(Clone, Copy)]
pub struct DivDenominatorMin;
//
impl Operation for DivDenominatorMin {
    const NAME: &str = "div_denominator_min";

    // One register for the upper bound
    fn aux_registers_regop(_input_registers: usize) -> usize {
        1
    }

    // Inputs are directly reduced into the accumulator, can use memory operands
    const AUX_REGISTERS_MEMOP: usize = 1 + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<Storage: InputsMut, const ILP: usize>(
        input_storage: Storage,
    ) -> impl Benchmark<Float = Storage::Element> {
        DivDenominatorMinBenchmark::<_, ILP> {
            input_storage,
            num_subnormals: None,
        }
    }
}

/// [`Benchmark`] of [`DivDenominatorMin`]
struct DivDenominatorMinBenchmark<Storage: InputsMut, const ILP: usize> {
    input_storage: Storage,
    num_subnormals: Option<usize>,
}
//
impl<Storage: InputsMut, const ILP: usize> Benchmark for DivDenominatorMinBenchmark<Storage, ILP> {
    type Float = Storage::Element;

    fn num_operations(&self) -> usize {
        operations::accumulated_len(&self.input_storage, ILP)
    }

    fn setup_inputs(&mut self, num_subnormals: usize) {
        assert!(num_subnormals <= self.input_storage.as_ref().len());
        self.num_subnormals = Some(num_subnormals);
    }

    #[inline]
    fn start_run(&mut self, rng: &mut impl Rng, inside_test: bool) -> Self::Run<'_> {
        generate_muldiv_inputs::<_, ILP>(
            self.num_subnormals
                .expect("Should have called setup_inputs first"),
            &mut self.input_storage,
            rng,
            inside_test,
            invert_normal,
            cancel_subnormal,
        );
        DivDenominatorMinRun {
            inputs: self.input_storage.freeze(),
            accumulators: operations::narrow_accumulators(rng, inside_test),
        }
    }

    type Run<'run>
        = DivDenominatorMinRun<Storage::Frozen<'run>, ILP>
    where
        Self: 'run;
}

/// [`BenchmarkRun`] of [`DivDenominatorMinBenchmark`]
struct DivDenominatorMinRun<Storage: Inputs, const ILP: usize> {
    inputs: Storage,
    accumulators: [Storage::Element; ILP],
}
//
impl<Storage: Inputs, const ILP: usize> BenchmarkRun for DivDenominatorMinRun<Storage, ILP> {
    type Float = Storage::Element;

    fn inputs(&self) -> &[Self::Float] {
        self.inputs.as_ref()
    }

    #[inline]
    fn integrate_inputs(&mut self) {
        // No need to hide inputs for this benchmark, the compiler can't exploit
        // its knowledge that inputs are being reused
        operations::integrate::<_, _, ILP, false>(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, true>,
            &mut self.inputs,
            integrate,
        );
    }

    #[inline]
    fn accumulators(&self) -> &[Storage::Element] {
        &self.accumulators
    }
}

/// Upper bound that is imposed on the accumulator value
fn upper_bound<T: FloatLike>() -> T {
    T::splat(4.0)
}

/// Normal input that cancels the effect of a previous normal input
fn invert_normal<T: FloatLike>(prev_input: T) -> T {
    T::splat(1.0) / prev_input
}

/// Normal input that cancels the effect of a previous subnormal input
fn cancel_subnormal<T: FloatLike>(input_after_subnormal: T) -> T {
    input_after_subnormal * upper_bound::<T>()
}

/// Integration step
#[inline]
fn integrate<T: FloatLike>(acc: T, elem: T) -> T {
    // - A normal input takes the accumulator from range [1/2; 2[ to
    //   range [1/4; 4[. If it is followed by another normal input,
    //   the input generation procedure forces it to be the inverse
    //   of that previous number, taking the accumulator back to its
    //   nominal range [1/2; 2[.
    // - If elem is subnormal, the MIN makes the output normal
    //   again, and equal to 4. It then stays at 4 for all further
    //   subnormal inputs, then the next normal input is a value in
    //   range [1/2; 2[ multiplied by 4, which is in range [2; 8[,
    //   and dividing the saturated accumulator value of 4 by this
    //   has the effect of shrinking the accumulator back to its
    //   nominal range [1/2; 2[.
    operations::hide_single_accumulator(acc / elem).fast_min(upper_bound())
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{
        inputs::generators::{
            muldiv::test_generate_muldiv_inputs, test_utils::target_and_num_subnormals,
        },
        operations::test_utils::NeedsNarrowAcc,
    };
    use proptest::prelude::*;

    /// Test the input generation logic
    fn test_generate_inputs<const ILP: usize>(
        target: &mut [f32],
        num_subnormals: usize,
    ) -> Result<(), TestCaseError> {
        test_generate_muldiv_inputs::<ILP>(
            num_subnormals,
            target,
            invert_normal,
            cancel_subnormal,
            integrate,
        )
    }
    //
    proptest! {
        #[test]
        fn generate_inputs_ilp1((mut target, num_subnormals) in target_and_num_subnormals(1)) {
            test_generate_inputs::<1>(&mut target, num_subnormals)?;
        }

        #[test]
        fn generate_inputs_ilp2((mut target, num_subnormals) in target_and_num_subnormals(2)) {
            test_generate_inputs::<2>(&mut target, num_subnormals)?;
        }

        #[test]
        fn generate_inputs_ilp3((mut target, num_subnormals) in target_and_num_subnormals(3)) {
            test_generate_inputs::<3>(&mut target, num_subnormals)?;
        }
    }

    // Test the `Operation` implementation
    common::test_scalar_operation!(DivDenominatorMin, NeedsNarrowAcc::FirstNormal);
}
