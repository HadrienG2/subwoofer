use common::{
    arch::HAS_MEMORY_OPERANDS,
    floats::{self, FloatLike},
    inputs::{self, Inputs, InputsMut},
    operations::{self, Benchmark, BenchmarkRun, Operation},
};
use rand::Rng;

/// FMA with possibly subnormal addend
#[derive(Clone, Copy)]
pub struct FmaAddend;
//
impl Operation for FmaAddend {
    const NAME: &str = "fma_addend";

    // One register for the multiplier, one for its inverse
    fn aux_registers_regop(_input_registers: usize) -> usize {
        2
    }

    // acc is not reused after FMA, so memory operands can be used. Without
    // memory operands, need to load elem into another register before FMA.
    const AUX_REGISTERS_MEMOP: usize = 2 + (!HAS_MEMORY_OPERANDS) as usize;

    fn make_benchmark<const ILP: usize>(input_storage: impl InputsMut) -> impl Benchmark {
        FmaAddendBenchmark::<_, ILP> {
            input_storage,
            num_subnormals: None,
        }
    }
}

/// [`Benchmark`] of [`FmaAddend`]
struct FmaAddendBenchmark<Storage: InputsMut, const ILP: usize> {
    input_storage: Storage,
    num_subnormals: Option<usize>,
}
//
impl<Storage: InputsMut, const ILP: usize> Benchmark for FmaAddendBenchmark<Storage, ILP> {
    fn num_operations(&self) -> usize {
        inputs::accumulated_len(&self.input_storage, ILP)
    }

    fn setup_inputs(&mut self, num_subnormals: usize) {
        self.num_subnormals = Some(num_subnormals);
    }

    #[inline]
    fn start_run(&mut self, rng: &mut impl Rng) -> Self::Run<'_> {
        let narrow = floats::narrow_sampler();
        let multiplier = narrow(rng);
        inputs::generate_input_pairs::<_, ILP>(
            &mut self.input_storage,
            rng,
            self.num_subnormals
                .expect("Should have called setup_inputs first"),
            /* ??? */
        );
        FmaAddendRun {
            inputs: self.input_storage.freeze(),
            accumulators: operations::narrow_accumulators(rng),
            multiplier,
            inv_multiplier: pessimize::hide(Storage::Element::splat(1.0) / multiplier),
        }
    }

    type Run<'run>
        = FmaAddendRun<Storage::Frozen<'run>, ILP>
    where
        Self: 'run;
}

/// [`BenchmarkRun`] of [`FmaAddendBenchmark`]
struct FmaAddendRun<Storage: Inputs, const ILP: usize> {
    inputs: Storage,
    accumulators: [Storage::Element; ILP],
    multiplier: Storage::Element,
    inv_multiplier: Storage::Element,
}
//
impl<Storage: Inputs, const ILP: usize> BenchmarkRun for FmaAddendRun<Storage, ILP> {
    type Float = Storage::Element;

    #[inline]
    fn integrate_inputs(&mut self) {
        let multiplier = self.multiplier;
        let inv_multiplier = self.inv_multiplier;
        operations::integrate_pairs::<_, _, ILP>(
            &mut self.accumulators,
            operations::hide_accumulators::<_, ILP, true>,
            &mut self.inputs,
            move |acc, [elem1, elem2]| {
                // - By flipping between a multiplier and its inverse, we ensure
                //   that the accumulator keeps the same order of magnitude over
                //   time.
                // - Normal input element generation is biased such that after
                //   adding a quantity to the accumulator, the same quantity is
                //   later subtracted back from it, with correct accounting of
                //   the effect of multipliers.
                //
                // TODO: Check need for finer-grained optimization barriers
                acc.mul_add(multiplier, elem1)
                    .mul_add(inv_multiplier, elem2)
            },
        );
    }

    #[inline]
    fn accumulators(&self) -> &[Self::Float] {
        &self.accumulators
    }
}
