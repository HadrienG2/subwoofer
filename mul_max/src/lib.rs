use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{generators::muldiv::generate_muldiv_inputs, Inputs, InputsMut},
    operations::{self, Benchmark, BenchmarkRun, Operation},
};
use rand::Rng;

/// MUL followed by MAX
#[derive(Clone, Copy)]
pub struct MulMax;
//
impl Operation for MulMax {
    const NAME: &str = "mul_max";

    // One register for the lower bound
    fn aux_registers_regop(_input_registers: usize) -> usize {
        1
    }

    // Inputs are directly reduced into the accumulator, can use memory operands
    const AUX_REGISTERS_MEMOP: usize = 1 + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<Storage: InputsMut, const ILP: usize>(
        input_storage: Storage,
    ) -> impl Benchmark<Float = Storage::Element> {
        MulMaxBenchmark::<_, ILP> {
            input_storage,
            num_subnormals: None,
        }
    }
}

/// [`Benchmark`] of [`MulMax`]
struct MulMaxBenchmark<Storage: InputsMut, const ILP: usize> {
    input_storage: Storage,
    num_subnormals: Option<usize>,
}
//
impl<Storage: InputsMut, const ILP: usize> Benchmark for MulMaxBenchmark<Storage, ILP> {
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
        MulMaxRun {
            inputs: self.input_storage.freeze(),
            accumulators: operations::narrow_accumulators(rng, inside_test),
        }
    }

    type Run<'run>
        = MulMaxRun<Storage::Frozen<'run>, ILP>
    where
        Self: 'run;
}

/// [`BenchmarkRun`] of [`MulMaxBenchmark`]
struct MulMaxRun<Storage: Inputs, const ILP: usize> {
    inputs: Storage,
    accumulators: [Storage::Element; ILP],
}
//
impl<Storage: Inputs, const ILP: usize> BenchmarkRun for MulMaxRun<Storage, ILP> {
    type Float = Storage::Element;

    fn inputs(&self) -> &[Self::Float] {
        self.inputs.as_ref()
    }

    #[inline]
    fn integrate_inputs(&mut self) {
        // No need to hide inputs for this benchmark, the compiler can't exploit
        // its knowledge that inputs are reused here
        operations::integrate::<_, _, ILP, false>(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, false>,
            &mut self.inputs,
            integrate,
        );
    }

    #[inline]
    fn accumulators(&self) -> &[Self::Float] {
        &self.accumulators
    }
}

/// Lower bound that is imposed on the accumulator value
#[inline]
fn lower_bound<T: FloatLike>() -> T {
    T::splat(0.25)
}

/// Normal input that cancels the effect of a previous normal input
fn invert_normal<T: FloatLike>(prev_input: T) -> T {
    T::splat(1.0) / prev_input
}

/// Normal input that cancels the effect of a previous subnormal input
fn cancel_subnormal<T: FloatLike>(input_after_subnormal: T) -> T {
    input_after_subnormal / lower_bound::<T>()
}

/// Integration step
#[inline]
fn integrate<T: FloatLike>(acc: T, elem: T) -> T {
    // - A normal input takes the accumulator from range [1/2; 2[ to
    //   range [1/4; 4[. If it is followed by another normal input,
    //   the input generation procedure forces it to be the inverse
    //   of that previous number, taking the accumulator back to its
    //   nominal range [1/2; 2[.
    // - If elem is subnormal, the MAX makes the output normal
    //   again, and equal to 1/4. It then stays at 1/4 for all
    //   further subnormal inputs. Then the next normal input is a
    //   value in range [1/2; 2[ divided by 1/4, and multiplying the
    //   saturated accumulator value of 1/4 by this takes the
    //   accumulator back to its nominal range [1/2; 2[.
    operations::hide_single_accumulator(acc * elem).fast_max(lower_bound())
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
    common::test_scalar_operation!(MulMax, NeedsNarrowAcc::FirstNormal, 20.0 * f32::EPSILON);
}
