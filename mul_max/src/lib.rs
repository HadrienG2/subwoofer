use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::FloatLike,
    inputs::{self, Inputs, InputsMut},
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

    fn make_benchmark<const ILP: usize>(input_storage: impl InputsMut) -> impl Benchmark {
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
    fn num_operations(&self) -> usize {
        operations::accumulated_len(&self.input_storage, ILP)
    }

    fn setup_inputs(&mut self, num_subnormals: usize) {
        self.num_subnormals = Some(num_subnormals);
    }

    #[inline]
    fn start_run(&mut self, rng: &mut impl Rng) -> Self::Run<'_> {
        inputs::generate_muldiv_inputs::<_, ILP>(
            &mut self.input_storage,
            rng,
            self.num_subnormals
                .expect("Should have called setup_inputs first"),
            |prev_input| Storage::Element::splat(1.0) / prev_input,
            |input_after_subnormal| input_after_subnormal / lower_bound::<Storage::Element>(),
        );
        MulMaxRun {
            inputs: self.input_storage.freeze(),
            accumulators: operations::narrow_accumulators(rng),
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

    #[inline]
    fn integrate_inputs(&mut self) {
        // No need to hide inputs for this benchmark, the compiler can't exploit
        // its knowledge that inputs are reused here
        let lower_bound = lower_bound::<Storage::Element>();
        operations::integrate::<_, _, ILP, false>(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, false>,
            &mut self.inputs,
            move |acc, elem| {
                // - A normal input takes the accumulator from range [1/2; 2[ to
                //   range [1/4; 4[. If it is followed by another normal input,
                //   the input generation procedure forces it to be the inverse
                //   of that previous number, taking the accumulator back to its
                //   nominal range [1/2; 2[.
                // - If elem is subnormal, the MAX makes the output normal
                //   again, and equal to 0.25. It then stays at 0.25 for all
                //   further subnormal inputs, then the next normal input is a
                //   value in range [1/2; 2[ divided by 0.25, and multiplying
                //   the saturated accumulator value of 0.25 by this has the
                //   effect of taking the accumulator back to its nominal range
                //   [1/2; 2[.
                operations::hide_single_accumulator(acc * elem).fast_max(lower_bound)
            },
        );
    }

    #[inline]
    fn accumulators(&self) -> &[Self::Float] {
        &self.accumulators
    }
}

/// Lower bound that is imposed on the accumulator value
fn lower_bound<T: FloatLike>() -> T {
    T::splat(0.25)
}
