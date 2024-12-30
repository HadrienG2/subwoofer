//! Benchmark inner loops and associated types

use crate::{
    floats::FloatLike,
    inputs::{FloatSequence, FloatSet},
};
use criterion::{measurement::WallTime, BenchmarkGroup, Throughput};
use rand::prelude::*;
use std::time::Instant;

/// Floating-point operation that can be benchmarked
pub trait Operation<T: FloatLike>: Copy {
    /// Name given to this operation in criterion benchmark outputs
    const NAME: &str;

    /// Minimal number of CPU registers used when processing in-register inputs,
    /// besides these inputs and the benchmark's inner accumulators.
    ///
    /// Used to decide which degrees of accumulation ILP are worth trying.
    ///
    /// This assumes the compiler does a perfect job at register allocation,
    /// which unfortunately is not always true, especially in the presence of
    /// optimization barriers like `pessimize::hide()` that may have the
    /// side-effect of artificially increasing register pressure. But we
    /// tolerate a bit of spill to the stack if there's a lot more arithmetic.
    fn aux_registers_regop(input_registers: usize) -> usize;

    /// Like `AUX_REGISTERS_MEMOP`, but for memory operands
    const AUX_REGISTERS_MEMOP: usize;

    /// Setup a benchmark with a certain degree of instruction-level parallelism
    fn make_benchmark<const ILP: usize>() -> impl Benchmark<Float = T>;
}
//
/// Microbenchmark of a certain [`Operation`]
pub trait Benchmark: Copy {
    /// Floating-point type that is being exercised
    type Float: FloatLike;

    /// Number of hardware operations under study that this benchmark will
    /// execute when processing a certain input dataset
    ///
    /// Most benchmarks consume one input element per operation, in which case
    /// this will be [`inputs.reused_len(ilp)`](FloatSet::reused_len) with `ilp`
    /// set to this benchmark's accumulation ILP. But beware of benchmarks like
    /// fma_full which consume two inputs per operation.
    fn num_operations<Inputs: FloatSet>(inputs: &Inputs) -> usize;

    /// Start a new benchmark run
    ///
    /// This is the point where a benchmark should initialize its internal
    /// state: accumulators, growth factors, etc. Ideally, the state should be
    /// fully regenerated from some suitable random distribution on each run to
    /// avoid measurement bias linked to multiple runs with a the same initial
    /// state. But if generating the state from an RNG is too expensive, a
    /// reused state can be passed through [`pessimize::hide()`] instead.
    ///
    /// You will typically want to initialize your accumulators to one of
    /// [`additive_accumulators()`] or [`normal_accumulators()`] here.
    ///
    /// Implementations should be marked `#[inline]` to give the compiler a more
    /// complete picture of the underlying data flow.
    fn begin_run(self, rng: impl Rng) -> Self;

    /// Integrate a new batch of inputs into the internal state
    ///
    /// Inputs must be returned in a state suitable for reuse on the next
    /// benchmark loop iteration. In most cases, they can just be left as-is,
    /// but if the compiler can hoist computations out of the benchmark loop
    /// under the knowledge that the inputs are always the same, then you will
    /// have to pass inputs through a [`pessimize::hide()`] barrier, which may
    /// come at the expense of less optimal codegen.
    ///
    /// If your benchmark sequentially feeds all inputs to a single accumulator,
    /// you may use the provided [`integrate_full()`] skeleton to implement this
    /// function. If your benchmark alternates between two different ways to
    /// integrate inputs, you may use the provided [`integrate_halves()`]
    /// skeleton to implement this function.
    ///
    /// In any case, you will also want to regularly pass accumulators through
    /// the provided [`hide_accumulators()`] function, otherwise the compiler
    /// will perform unwanted autovectorization and you will be unable to
    /// compare scalar vs SIMD overhead.
    ///
    /// Implementations must be marked `#[inline]` as this function will be
    /// called as part of the timed benchmark run.
    fn integrate_inputs<Inputs>(&mut self, inputs: &mut Inputs)
    where
        Inputs: FloatSequence<Element = Self::Float>;

    /// End a benchmark run by making the compiler think the output is used
    ///
    /// It is normally enough to pass the benchmark's internal accumulators
    /// through the provided [`consume_accumulators()`] function.
    ///
    /// Implementations should be marked `#[inline]` to give the compiler a more
    /// complete picture of the underlying data flow.
    fn consume_outputs(self);
}

/// Measure the performance of a [`Benchmark`] on certain inputs
#[inline(never)] // Faster build + easier profiling
pub fn run_benchmark<B: Benchmark>(
    benchmark: B,
    group: &mut BenchmarkGroup<WallTime>,
    mut inputs: impl FloatSet<Element = B::Float>,
    input_name: String,
    mut rng: impl Rng,
) {
    group.throughput(Throughput::Elements(B::num_operations(&inputs) as u64));
    group.bench_function(input_name, move |b| {
        b.iter_custom(|iters| {
            // For each benchmark iteration batch, we ensure that...
            //
            // - The compiler cannot leverage the fact that the initial
            //   accumulators are always the same in order to eliminate
            //   "redundant" computations across iteration batches.
            // - Inputs are randomly reordered from one batch to another, which
            //   will avoid input order-related hardware bias if criterion runs
            //   enough batches (as it does in its default configuration).
            let mut benchmark = benchmark.begin_run(&mut rng);
            let mut inputs = inputs.make_sequence(&mut rng);

            // Timed region, this is the danger zone where inlining and compiler
            // optimizations must be reviewed very carefully.
            let start = Instant::now();
            for _ in 0..iters {
                benchmark.integrate_inputs(&mut inputs);
            }
            let elapsed = start.elapsed();

            // Tell the compiler that the accumulated results are used so it
            // doesn't delete the computation
            benchmark.consume_outputs();
            elapsed
        })
    });
}

/// Initialize accumulators with a positive value that is suitable as the
/// starting point of an additive random walk of ~unity step.
///
/// The chosen initial value takes a half-way compromise between two objectives:
///
/// - When many positive values are subtracted, it should not get close to zero
/// - When many positive values are added, it should not saturate to a maximum
#[inline]
pub fn additive_accumulators<T: FloatLike, const ILP: usize>(mut rng: impl Rng) -> [T; ILP] {
    let normal_sampler = T::normal_sampler();
    std::array::from_fn::<_, ILP, _>(|_| {
        T::splat(2.0f32.powi((T::MANTISSA_DIGITS / 2) as i32)) * normal_sampler(&mut rng)
    })
}

/// Initialize accumulators with a positive value close to 1.0
///
/// This is the right default for most benchmarks, except those that perform an
/// additive random walk.
#[inline]
pub fn normal_accumulators<T: FloatLike, const ILP: usize>(mut rng: impl Rng) -> [T; ILP] {
    let normal_sampler = T::normal_sampler();
    std::array::from_fn::<_, ILP, _>(|_| normal_sampler(&mut rng))
}

/// Benchmark skeleton that processes the full input homogeneously
#[inline]
pub fn integrate_full<
    T: FloatLike,
    Inputs: FloatSequence<Element = T>,
    const ILP: usize,
    const HIDE_INPUTS: bool,
>(
    accumulators: &mut [T; ILP],
    inputs: &mut Inputs,
    mut iter: impl Copy + FnMut(T, T) -> T,
) {
    let inputs_slice = inputs.as_ref();
    if Inputs::IS_REUSED {
        if HIDE_INPUTS {
            // When we need to hide inputs, we flip the order of iteration over
            // inputs and accumulators so that each set of inputs is fully
            // processed before we switch accumulators.
            //
            // This minimizes the number of input optimization barriers, at the
            // expense of making it harder for the CPU to extract ILP if the
            // backend doesn't reorder instructions across the optimization
            // barrier, because the CPU frontend must now dive through N
            // mentions of an accumulator before reaching mentions of the next
            // accumulator (we lose the "jam" part of unroll & jam).
            for acc in accumulators.iter_mut() {
                for &elem in inputs.as_ref() {
                    *acc = iter(*acc, elem);
                }
                <Inputs as FloatSequence>::hide_inplace(inputs);
            }
            hide_accumulators(accumulators);
        } else {
            for &elem in inputs_slice {
                for acc in accumulators.iter_mut() {
                    *acc = iter(*acc, elem);
                }
                hide_accumulators(accumulators);
            }
        }
    } else {
        let chunks = inputs_slice.chunks_exact(ILP);
        let remainder = chunks.remainder();
        for chunk in chunks {
            for (&elem, acc) in chunk.iter().zip(accumulators.iter_mut()) {
                *acc = iter(*acc, elem);
            }
            hide_accumulators(accumulators);
        }
        for (&elem, acc) in remainder.iter().zip(accumulators.iter_mut()) {
            *acc = iter(*acc, elem);
        }
    }
}

/// Benchmark skeleton that treats each half of the input differently
#[inline]
pub fn integrate_halves<T: FloatLike, Inputs: FloatSequence<Element = T>, const ILP: usize>(
    accumulators: &mut [T; ILP],
    inputs: &Inputs,
    mut low_iter: impl Copy + FnMut(T, T) -> T,
    mut high_iter: impl Copy + FnMut(T, T) -> T,
) {
    if Inputs::IS_REUSED {
        // Otherwise, we just do the same as usual, but with input reuse
        let inputs_slice = inputs.as_ref();
        let (low_inputs, high_inputs) = inputs_slice.split_at(inputs_slice.len() / 2);
        assert_eq!(low_inputs.len(), high_inputs.len());
        for (&low_elem, &high_elem) in low_inputs.iter().zip(high_inputs) {
            for acc in accumulators.iter_mut() {
                *acc = low_iter(*acc, low_elem);
                *acc = high_iter(*acc, high_elem);
            }
            hide_accumulators(accumulators);
        }
    } else {
        let inputs_slice = inputs.as_ref();
        let (low_inputs, high_inputs) = inputs_slice.split_at(inputs_slice.len() / 2);
        let low_chunks = low_inputs.chunks_exact(ILP);
        let high_chunks = high_inputs.chunks_exact(ILP);
        let low_remainder = low_chunks.remainder();
        let high_remainder = high_chunks.remainder();
        for (low_chunk, high_chunk) in low_chunks.zip(high_chunks) {
            for ((&low_elem, &high_elem), acc) in low_chunk
                .iter()
                .zip(high_chunk.iter())
                .zip(accumulators.iter_mut())
            {
                *acc = low_iter(*acc, low_elem);
                *acc = high_iter(*acc, high_elem);
            }
            hide_accumulators(accumulators);
        }
        for (&low_elem, acc) in low_remainder.iter().zip(accumulators.iter_mut()) {
            *acc = low_iter(*acc, low_elem);
        }
        for (&high_elem, acc) in high_remainder.iter().zip(accumulators.iter_mut()) {
            *acc = high_iter(*acc, high_elem);
        }
    }
}

/// Minimal optimization barrier needed to avoid accumulator autovectorization
///
/// If we accumulate data naively, LLVM will figure out that we're aggregating
/// arrays of inputs into identically sized arrays of accumulators and helpfully
/// autovectorize the accumulation. We do not want this optimization here
/// because it gets in the way of us separately studying the hardware overhead
/// of scalar and SIMD operations.
///
/// Therefore, we send accumulators to opaque inline ASM via `pessimize::hide()`
/// after ~each accumulation iteration so that LLVM is forced to run the
/// computations on the actual specified data type, or more precisely decides to
/// do so because the overhead of assembling data in wide vectors then
/// re-splitting it into narrow vectors is higher than the performance gain of
/// autovectorizing.
///
/// Unfortunately, such frequent optimization barriers come with side effects
/// like useless register-register moves and a matching increase in FP register
/// pressure. To reduce and ideally fully eliminate the impact, it is better to
/// send only a subset of the accumulators through the optimization barrier, and
/// to refrain from using it altogether in situations where autovectorization is
/// not known to be possible. This is what this function does.
///
/// For better codegen, you can also abandon the convenience of the `-C
/// target-cpu=native` configuration that is enabled in the `.cargo/config.toml`
/// file, and instead separately benchmark each type in a dedicated
/// configuration that does not enable unneeded wider vector instructions. On
/// x86_64, that would be e.g. `-C target_feature=+avx` for f32x08 and f64x04,
/// and nothing for SSE and scalar types (it is not legal to disable SSE for
/// better scalar codegen on x86_64).
#[inline]
pub fn hide_accumulators<T: FloatLike, const ILP: usize>(accumulators: &mut [T; ILP]) {
    // No need for pessimize::hide() optimization barriers if this accumulation
    // cannot be autovectorized because e.g. we are operating at the maximum
    // hardware-supported SIMD vector width.
    let Some(min_vector_ilp) = T::MIN_VECTORIZABLE_ILP else {
        return;
    };

    // Otherwise apply pessimize::hide() to a set of accumulators which is large
    // enough that there aren't enough remaining non-hidden accumulators left
    // for the compiler to perform autovectorization.
    //
    // If rustc/LLVM ever becomes crazy enough to autovectorize with masked SIMD
    // operations, then this -1 may need to become a /2 or worse, depending on
    // how efficient the compiler estimates that masked SIMD operations are on
    // the CPU target that the benchmark will run on. It's debatable whether
    // this is best handled here or in the definition of MIN_VECTORIZABLE_ILP.
    let max_elided_barriers = min_vector_ilp.get() - 1;
    let min_hidden_accs = ILP.saturating_sub(max_elided_barriers);
    for acc in accumulators.iter_mut().take(min_hidden_accs) {
        let old_acc = *acc;
        let new_acc = pessimize::hide::<T>(old_acc);
        *acc = new_acc;
    }
}

/// Consume accumulators at the end of a benchmark run
#[inline]
pub fn consume_accumulators<T: FloatLike, const ILP: usize>(accumulators: [T; ILP]) {
    for acc in accumulators {
        pessimize::consume::<T>(acc);
    }
}
