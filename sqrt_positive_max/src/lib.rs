use common::{
    floats::FloatLike,
    inputs::{generators::max::generate_max_inputs, InputKind, Inputs, InputsMut},
    operations::{self, Benchmark, BenchmarkRun, Operation},
};
use rand::prelude::*;

/// Square root of positive numbers, followed by MAX
///
/// Square roots of negative numbers may or may not be emulated in software.
/// They are thus not a good candidate for CPU microbenchmarking.
#[derive(Clone, Copy)]
pub struct SqrtPositiveMax;
//
impl Operation for SqrtPositiveMax {
    const NAME: &str = "sqrt_positive_max";

    // A square root must go to a temporary before use...
    fn aux_registers_regop(input_registers: usize) -> usize {
        // ...and unfortunately, it looks like the input optimization barrier
        // that we apply to avoid sqrt-hoisting has the side-effect of
        // pessimizing this into a full doubling of register pressure.
        input_registers
    }

    // ...but architectures without memory operands can reuse the register that
    // was used to load the input in order to hold the square root.
    const AUX_REGISTERS_MEMOP: usize = 1;

    fn make_benchmark<const ILP: usize>(input_storage: impl InputsMut) -> impl Benchmark {
        SqrtPositiveMaxBenchmark::<_, ILP> {
            input_storage,
            num_subnormals: None,
        }
    }
}

/// [`Benchmark`] of [`SqrtPositiveMax`]
struct SqrtPositiveMaxBenchmark<Storage: InputsMut, const ILP: usize> {
    input_storage: Storage,
    num_subnormals: Option<usize>,
}
//
impl<Storage: InputsMut, const ILP: usize> Benchmark for SqrtPositiveMaxBenchmark<Storage, ILP> {
    fn num_operations(&self) -> usize {
        operations::accumulated_len(&self.input_storage, ILP)
    }

    fn setup_inputs(&mut self, num_subnormals: usize) {
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
        SqrtPositiveMaxRun {
            inputs: self.input_storage.freeze(),
            accumulators: operations::normal_accumulators(rng),
        }
    }

    type Run<'run>
        = SqrtPositiveMaxRun<Storage::Frozen<'run>, ILP>
    where
        Self: 'run;
}

/// [`BenchmarkRun`] of [`SqrtPositiveMax`]
struct SqrtPositiveMaxRun<I: Inputs, const ILP: usize> {
    inputs: I,
    accumulators: [I::Element; ILP],
}
//
impl<I: Inputs, const ILP: usize> BenchmarkRun for SqrtPositiveMaxRun<I, ILP> {
    type Float = I::Element;

    #[inline]
    fn integrate_inputs(&mut self) {
        let hide_accumulators = operations::hide_accumulators::<_, ILP, false>;
        let iter = |acc: I::Element, elem: I::Element| {
            // MAX is unaffected by the order of magnitude of inputs, and SQRT
            // transforms any normal number into another normal number without
            // any possibility of overflow or underflow, so this benchmark
            // behaves homogeneously no matter what the order of magnitude of
            // its normal inputs is.
            acc.fast_max(operations::hide_single_accumulator(elem.sqrt()))
        };
        match I::KIND {
            InputKind::ReusedRegisters { .. } => {
                // Need to hide reused register inputs, so that the compiler doesn't
                // abusively factor out the "redundant" square root computations and
                // reuse their result for all accumulators. In fact, if the compiler
                // were smarter, it would even be allowed to reuse these square
                // roots during the whole outer loop of run_benchmark...
                operations::integrate::<_, _, ILP, true>(
                    &mut self.accumulators,
                    hide_accumulators,
                    &mut self.inputs,
                    iter,
                )
            }
            InputKind::Memory => {
                // Memory inputs do not need to be hidden because each accumulator
                // gets its own input substream (preventing square root reuse during
                // the inner loop over accumulators), and current LLVM is not crazy
                // enough to precompute square roots for a whole arbitrarily large
                // and dynamically-sized batch of input data.
                operations::integrate::<_, _, ILP, false>(
                    &mut self.accumulators,
                    hide_accumulators,
                    &mut self.inputs,
                    iter,
                )
            }
        }
    }

    #[inline]
    fn accumulators(&self) -> &[I::Element] {
        &self.accumulators
    }
}
