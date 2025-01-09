use common::{
    arch::{HAS_HARDWARE_NEGATED_FMA, HAS_MEMORY_OPERANDS},
    floats::{self, FloatLike},
    inputs::{self, Inputs, InputsMut},
    operations::{self, Benchmark, BenchmarkRun, Operation},
};
use rand::Rng;

/// FMA with possibly subnormal multiplier, used in an ADD/SUB cycle
#[derive(Clone, Copy)]
pub struct FmaMultiplierBidi;
//
impl Operation for FmaMultiplierBidi {
    const NAME: &str = "fma_multiplier_bidi";

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

    fn make_benchmark<const ILP: usize>(input_storage: impl InputsMut) -> impl Benchmark {
        FmaMultiplierBidiBenchmark::<_, ILP> {
            input_storage,
            num_subnormals: None,
        }
    }
}

/// [`Benchmark`] of [`FmaMultiplierBidi`]
struct FmaMultiplierBidiBenchmark<Storage: InputsMut, const ILP: usize> {
    input_storage: Storage,
    num_subnormals: Option<usize>,
}
//
impl<Storage: InputsMut, const ILP: usize> Benchmark for FmaMultiplierBidiBenchmark<Storage, ILP> {
    fn num_operations(&self) -> usize {
        inputs::accumulated_len(&self.input_storage, ILP)
    }

    fn setup_inputs(&mut self, num_subnormals: usize) {
        self.num_subnormals = Some(num_subnormals);
    }

    #[inline]
    fn start_run(&mut self, rng: &mut impl Rng) -> Self::Run<'_> {
        inputs::generate_add_inputs::<_, ILP>(
            &mut self.input_storage,
            rng,
            self.num_subnormals
                .expect("Should have called setup_inputs first"),
        );
        let narrow = floats::narrow_sampler();
        FmaMultiplierBidiRun {
            inputs: self.input_storage.freeze(),
            accumulators: operations::narrow_accumulators(rng),
            multiplier: narrow(rng),
        }
    }

    type Run<'run>
        = FmaMultiplierBidiRun<Storage::Frozen<'run>, ILP>
    where
        Self: 'run;
}

/// [`BenchmarkRun`] of [`FmaMultiplierBidi`]
struct FmaMultiplierBidiRun<I: Inputs, const ILP: usize> {
    inputs: I,
    accumulators: [I::Element; ILP],
    multiplier: I::Element,
}
//
impl<Storage: Inputs, const ILP: usize> BenchmarkRun for FmaMultiplierBidiRun<Storage, ILP> {
    type Float = Storage::Element;

    #[inline]
    fn integrate_inputs(&mut self) {
        // Overall, this is just the add benchmark with a step size that is at
        // most 2x smaller/larger, which fundamentally doesn't change much
        let multiplier = self.multiplier;
        operations::integrate::<_, _, ILP, false>(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, false>,
            &mut self.inputs,
            move |acc, elem| elem.mul_add(multiplier, acc),
        );
    }

    #[inline]
    fn accumulators(&self) -> &[Storage::Element] {
        &self.accumulators
    }
}
