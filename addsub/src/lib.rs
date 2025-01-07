use common::{
    arch::HAS_MEMORY_OPERANDS,
    inputs::{self, AddSubInputs, Inputs, InputsMut},
    operations::{self, Benchmark, BenchmarkRun, Operation},
};
use rand::Rng;

/// ADD/SUB cycle
#[derive(Clone, Copy)]
pub struct AddSub;
//
impl Operation for AddSub {
    const NAME: &str = "addsub";

    fn aux_registers_regop(_input_registers: usize) -> usize {
        0
    }

    // Inputs are directly reduced into the accumulator, can use memory operands
    const AUX_REGISTERS_MEMOP: usize = (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>(input_storage: impl InputsMut) -> impl Benchmark {
        AddSubBenchmark::<_, ILP> {
            input_storage: AddSubInputs::from(input_storage),
            num_subnormals: None,
        }
    }
}

/// [`Benchmark`] of [`AddSub`]
struct AddSubBenchmark<Storage: InputsMut, const ILP: usize> {
    input_storage: AddSubInputs<Storage>,
    num_subnormals: Option<usize>,
}
//
impl<Storage: InputsMut, const ILP: usize> Benchmark for AddSubBenchmark<Storage, ILP> {
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
        AddSubRun {
            inputs: self.input_storage.freeze(),
            accumulators: operations::additive_accumulators(rng),
        }
    }

    type Run<'run>
        = AddSubRun<Storage::Frozen<'run>, ILP>
    where
        Self: 'run;
}

/// [`BenchmarkRun`] of [`AddSubBenchmark`]
struct AddSubRun<Storage: Inputs, const ILP: usize> {
    inputs: AddSubInputs<Storage>,
    accumulators: [Storage::Element; ILP],
}
//
impl<Storage: Inputs, const ILP: usize> BenchmarkRun for AddSubRun<Storage, ILP> {
    type Float = Storage::Element;

    #[inline]
    fn integrate_inputs(&mut self) {
        operations::integrate_halves(
            &mut self.accumulators,
            &self.inputs,
            |acc, elem| acc + elem,
            |acc, elem| acc - elem,
        );
    }

    #[inline]
    fn accumulators(&self) -> &[Storage::Element] {
        &self.accumulators
    }
}
