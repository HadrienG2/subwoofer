use common::{
    arch::{HAS_HARDWARE_NEGATED_FMA, HAS_MEMORY_OPERANDS},
    floats::FloatLike,
    inputs::{self, AddSubInputs, Inputs, InputsMut},
    operations::{self, Benchmark, BenchmarkRun, Operation},
};
use rand::Rng;

/// FMA with possibly subnormal multiplier, used in an ADD/SUB pattern
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
            input_storage: AddSubInputs::from(input_storage),
            num_subnormals: None,
        }
    }
}

/// [`Benchmark`] of [`FmaMultiplierBidi`]
struct FmaMultiplierBidiBenchmark<Storage: InputsMut, const ILP: usize> {
    input_storage: AddSubInputs<Storage>,
    num_subnormals: Option<usize>,
}
//
impl<Storage: InputsMut, const ILP: usize> Benchmark for FmaMultiplierBidiBenchmark<Storage, ILP> {
    fn num_operations(&self) -> usize {
        inputs::accumulated_len(&self.input_storage, ILP)
    }

    const SUBNORMAL_INPUT_GRANULARITY: usize = 2;

    fn setup_inputs(&mut self, rng: &mut impl Rng, num_subnormals: usize) {
        // Decide whether to generate inputs now or every run
        if operations::should_generate_every_run(&self.input_storage, num_subnormals, ILP, 2) {
            self.num_subnormals = Some(num_subnormals);
        } else {
            self.input_storage.generate(rng, num_subnormals);
            self.num_subnormals = None;
        }
    }

    #[inline]
    fn start_run(&mut self, rng: &mut impl Rng) -> Self::Run<'_> {
        // Generate inputs if needed, otherwise just shuffle them around
        if let Some(num_subnormals) = self.num_subnormals {
            self.input_storage.generate(rng, num_subnormals);
        } else {
            self.input_storage.shuffle(rng);
        }

        // This is just a random additive walk of ~unity or subnormal step, so
        // given a high enough starting point, an initially normal accumulator
        // should stay in the normal range forever.
        let normal_sampler = Storage::Element::normal_sampler();
        FmaMultiplierBidiRun {
            inputs: self.input_storage.freeze(),
            accumulators: operations::additive_accumulators(rng),
            multiplier: normal_sampler(rng),
        }
    }

    type Run<'run>
        = FmaMultiplierBidiRun<Storage::Frozen<'run>, ILP>
    where
        Self: 'run;
}

/// [`BenchmarkRun`] of [`FmaMultiplierBidi`]
struct FmaMultiplierBidiRun<Storage: Inputs, const ILP: usize> {
    inputs: AddSubInputs<Storage>,
    accumulators: [Storage::Element; ILP],
    multiplier: Storage::Element,
}
//
impl<Storage: Inputs, const ILP: usize> BenchmarkRun for FmaMultiplierBidiRun<Storage, ILP> {
    type Float = Storage::Element;

    #[inline]
    fn integrate_inputs(&mut self) {
        // Overall, this is just the addsub benchmark with a step size that is
        // at most 2x smaller/larger, which fundamentally doesn't change much
        let multiplier = self.multiplier;
        operations::integrate_halves::<_, _, ILP>(
            &mut self.accumulators,
            &self.inputs,
            move |acc, elem| elem.mul_add(multiplier, acc),
            move |acc, elem| elem.mul_add(-multiplier, acc),
        );
    }

    #[inline]
    fn accumulators(&self) -> &[Storage::Element] {
        &self.accumulators
    }
}
