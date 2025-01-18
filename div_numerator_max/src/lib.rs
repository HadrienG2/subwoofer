use common::{
    arch::{ALLOWS_DIV_MEMORY_NUMERATOR, ALLOWS_DIV_OUTPUT_DENOMINATOR},
    floats::FloatLike,
    inputs::{self, Inputs, InputsMut},
    operations::{self, Benchmark, BenchmarkRun, Operation},
};
use rand::Rng;

/// DIV with a possibly subnormal numerator, followed by MAX
#[derive(Clone, Copy)]
pub struct DivNumeratorMax;
//
impl Operation for DivNumeratorMax {
    const NAME: &str = "div_numerator_max";

    // One register for the lower bound + a temporary on all CPU ISAs where DIV
    // cannot emit its output to the register that holds the denominator
    fn aux_registers_regop(_input_registers: usize) -> usize {
        1 + ALLOWS_DIV_OUTPUT_DENOMINATOR as usize
    }

    // Still need the lower bound register. But the question is, can we use a
    // memory input for the numerator of a division?
    const AUX_REGISTERS_MEMOP: usize = 1 + (!ALLOWS_DIV_MEMORY_NUMERATOR) as usize;

    fn make_benchmark<const ILP: usize>(input_storage: impl InputsMut) -> impl Benchmark {
        DivNumeratorMaxBenchmark::<_, ILP> {
            input_storage,
            num_subnormals: None,
        }
    }
}

/// [`Benchmark`] of [`DivNumeratorMax`]
struct DivNumeratorMaxBenchmark<Storage: InputsMut, const ILP: usize> {
    input_storage: Storage,
    num_subnormals: Option<usize>,
}
//
impl<Storage: InputsMut, const ILP: usize> Benchmark for DivNumeratorMaxBenchmark<Storage, ILP> {
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
            |prev_input| prev_input,
            |input_after_subnormal| input_after_subnormal * lower_bound::<Storage::Element>(),
        );
        DivNumeratorMaxRun {
            inputs: self.input_storage.freeze(),
            accumulators: operations::narrow_accumulators(rng),
        }
    }

    type Run<'run>
        = DivNumeratorMaxRun<Storage::Frozen<'run>, ILP>
    where
        Self: 'run;
}

/// [`BenchmarkRun`] of [`DivNumeratorMaxBenchmark`]
struct DivNumeratorMaxRun<Storage: Inputs, const ILP: usize> {
    inputs: Storage,
    accumulators: [Storage::Element; ILP],
}
//
impl<Storage: Inputs, const ILP: usize> BenchmarkRun for DivNumeratorMaxRun<Storage, ILP> {
    type Float = Storage::Element;

    #[inline]
    fn integrate_inputs(&mut self) {
        // No need to hide inputs for this benchmark, the compiler can't exploit
        // its knowledge that inputs are being reused here
        let lower_bound = lower_bound::<Storage::Element>();
        operations::integrate::<_, _, ILP, false>(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, false>,
            &mut self.inputs,
            move |acc, elem| {
                // - A normal input takes the accumulator from range [1/2; 2[ to
                //   range [1/4; 4[. If it is followed by another normal input,
                //   the input generation procedure forces it to be a copy of
                //   that previous number, so we're computing
                //   input1/(input1/acc) = acc, taking the accumulator back to
                //   its nominal range [1/2; 2[.
                // - If elem is subnormal, the MAX makes the output normal
                //   again, and equal to lower bound 1/4. It then stays at 1/4
                //   for all further subnormal inputs, then the next normal
                //   input is a value in range [1/2; 2[ multiplied by 1/4, and
                //   dividing this by the saturated accumulator value of 1/4 has
                //   the effect of bringing the accumulator back to its nominal
                //   range [1/2; 2[.
                operations::hide_single_accumulator(elem / acc).fast_max(lower_bound)
            },
        );
    }

    #[inline]
    fn accumulators(&self) -> &[Storage::Element] {
        &self.accumulators
    }
}

/// Lower bound that is imposed on the accumulator value
fn lower_bound<T: FloatLike>() -> T {
    T::splat(0.25)
}
