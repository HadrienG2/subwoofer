//! Benchmark inner loops and associated types

use crate::{
    floats::FloatLike,
    inputs::{Inputs, InputsMut},
};
use rand::prelude::*;

/// Floating-point operation that can be benchmarked
pub trait Operation: Copy {
    /// Name given to this operation in criterion benchmark outputs
    const NAME: &str;

    /// Minimal number of CPU registers used when processing in-register inputs,
    /// besides these inputs and the benchmark's inner accumulators.
    ///
    /// Used to decide which degrees of accumulation ILP are worth trying.
    ///
    /// This assumes the compiler does a perfect job at register allocation,
    /// which unfortunately is not always true, especially in the presence of
    /// optimization barriers like [`pessimize::hide()`] that may have the
    /// side-effect of artificially increasing register pressure. But we
    /// tolerate a bit of spill to the stack if there's a lot more arithmetic.
    fn aux_registers_regop(input_registers: usize) -> usize;

    /// Like `aux_registers_regop`, but for memory operands
    const AUX_REGISTERS_MEMOP: usize;

    /// Setup a benchmark with some input storage & accumulation parallelism
    fn make_benchmark<const ILP: usize>(input_storage: impl InputsMut) -> impl Benchmark;
}
//
/// Microbenchmark of a certain [`Operation`]
pub trait Benchmark {
    /// Number of hardware operations under study that this benchmark will
    /// execute when processing the previously specified input dataset
    ///
    /// Most benchmarks consume one input element per computed operation, in
    /// which case this will be [`inputs::accumulated_len(&input_storage,
    /// ilp)`](crate::inputs::accumulated_len), with `input_storage` pointing to
    /// the previously specified input storage and `ilp` set to this benchmark's
    /// degree of accumulation instruction-level parallelism.
    ///
    /// However, beware of benchmarks like `fma_full` which consume multiple
    /// input elements per computed operation.
    fn num_operations(&self) -> usize;

    /// Granularity of the amount of subnormal numbers in the input dataset
    ///
    /// The amount of subnormal numbers in the input dataset should be a
    /// multiple of this granularity. Otherwise it will be rounded up or down to
    /// a multiple of this granularity.
    const SUBNORMAL_INPUT_GRANULARITY: usize;

    /// Specify the desired share of subnormal numbers in the input dataset.
    ///
    /// The specified `num_subnormals` subnormal count must be smaller than or
    /// equal to the length of the underlying input storage.
    ///
    /// It should also be a multiple of `SUBNORMAL_INPUT_GRANULARITY`, otherwise
    /// it will be rounded up or down to such a multiple.
    ///
    /// The benchmark can generate an input dataset as soon as this function has
    /// been called. Whether it should do so right away during the call to this
    /// function is debatable:
    ///
    /// - Small input sets should be regenerated on each call to `start_run()`
    ///   in order to ensure that over Sufficiently Many runs, the benchmark is
    ///   exposed to a fair share of all possible inputs. In that case, the
    ///   value of `num_subnormals` should merely be cached by `setup_inputs()`
    ///   for use by `start_run()`.
    /// - Larger input sets are sufficiently representative of the space of all
    ///   possible inputs that regenerating them on each benchmark run is
    ///   overkill (and time-consuming). In that case, generating a single set
    ///   of inputs during `setup_inputs()` and reusing it for each subsequent
    ///   call to `start_run()` is fine. However, the position of subnormal
    ///   numbers within the input data stream may still affect the performance
    ///   of the CPU's floating-point primitives, so it remains a good idea to
    ///   at least randomly shuffle inputs on each call to `start_run()`.
    ///
    /// Implementations of `Benchmark` are advised to use the provided
    /// [`should_generate_every_run()`] utility to decide whether they should
    /// fully regenerate their input data set on every run, or generate it
    /// inside of `setup_inputs()` merely shuffle it inside of `start_run()`.
    fn setup_inputs(&mut self, rng: &mut impl Rng, num_subnormals: usize);

    /// Start a new benchmark run (= repeated passes through identical inputs)
    ///
    /// It is guaranteed that method `setup_inputs()` will be called at least
    /// once before this function is first called.
    ///
    /// Ideally, all benchmark-relevant state should be fully regenerated from
    /// some suitable random distribution on each run, in order to avoid
    /// measurement bias linked to a particular choice of benchmark state.
    ///
    /// But if generating the state from an RNG has a cost that is comparable to
    /// or higher than that of running the benchmark via `integrate_inputs()`,
    /// then the following alternatives should be considered:
    ///
    /// - Keep the same values, but randomly shuffle them. This is good enough
    ///   for relatively large sets of random values, like in-RAM input
    ///   datasets, where there is no meaningful difference between a randomly
    ///   reordered dataset and a fully regenerated one.
    /// - Keep the same values, just pass them through [`pessimize::hide()`] so
    ///   the compiler doesn't know that they are the same. This should be a
    ///   last-resort option, as it maximizes measurement bias.
    ///
    /// You will normally want to initialize the accumulators to one of
    /// [`additive_accumulators()`] or [`normal_accumulators()`] in this
    /// function. The input storage of the resulting [`BenchmarkRun`] should be
    /// derived from that of this benchmark through
    /// [`clone_or_reborrow()`](InputStorage::clone_or_reborrow).
    ///
    /// Implementations should be marked `#[inline]` to give the compiler a more
    /// complete picture of the accumulator data flow.
    fn start_run(&mut self, rng: &mut impl Rng) -> Self::Run<'_>;

    /// State of an ongoing execution of this benchmark
    type Run<'run>: BenchmarkRun
    where
        Self: 'run;
}
//
/// Ongoing execution of a certain [`Benchmark`]
pub trait BenchmarkRun {
    /// Floating-point type that is being exercised
    type Float: FloatLike;

    /// Integrate the input dataset into the inner accumulators
    ///
    /// This will be repeatedly called, for an unknown number of times, so...
    ///
    /// - The sequence of inputs should be chosen such that after iterating over
    ///   it, the benchmark accumulators either end up back at their starting
    ///   point, or at least converge to a normal limit with properties that are
    ///   similar to those of the starting point.
    /// - Inputs should be passed through a strategically placed
    ///   [`hide_inplace()`](InputStorage::hide_inplace) optimization barrier if
    ///   there is any chance that the compiler may optimize under the knowledge
    ///   that we are repeatedly running over the same inputs.
    ///
    /// If your benchmark sequentially feeds all inputs to a single accumulator,
    /// then you may use the provided [`integrate_full()`] skeleton to implement
    /// this function. If your benchmark alternates between two different ways
    /// to integrate inputs into accumulators, then you may use the provided
    /// [`integrate_halves()`] skeleton to implement this function.
    ///
    /// In any case, you will also want to regularly pass accumulators through
    /// the provided [`hide_accumulators()`] function, otherwise the compiler
    /// will perform unwanted autovectorization and you will be unable to
    /// compare scalar vs SIMD overhead. In more extreme cases where the
    /// compiler is _strongly_ convinced that autovectorization is worthwhile,
    /// you may even need to pass intermediary results through
    /// [`hide_single_accumulator()`].
    ///
    /// Implementations must be marked `#[inline]` as this function will be
    /// called as part of the timed benchmark run.
    fn integrate_inputs(&mut self);

    /// Access the benchmark's data accumulators
    ///
    /// This is used to mark the output of a benchmark as used, so that the
    /// compiler doesn't delete the computation.
    ///
    /// Implementations should be marked `#[inline]` to give the compiler a more
    /// complete picture of the accumulator data flow.
    fn accumulators(&self) -> &[Self::Float];
}

/// Truth that a benchmark should fully regenerate its input data set on every
/// run, rather than generate it inside of `setup_inputs()` and merely reshuffle
/// it inside of `start_run()`.
///
/// `num_subnormals` should be set to the number of subnormal values in the data
/// stream. This number should be exact i.e. it should be a multiple of
/// `SUBNORMAL_INPUT_GRANULARITY` for the corresponding benchmark.
///
/// `ilp` should be set to the number of accumulators that the benchmark uses
/// (i.e. its degree of Instruction-Level-Parallelism).
///
/// `substreams_per_accumulator` should be set to the number of data streams
/// that each accumulator feeds from.
pub fn should_generate_every_run<I: Inputs>(
    data: &I,
    num_subnormals: usize,
    ilp: usize,
    substreams_per_accumulator: usize,
) -> bool {
    // Figure out which values are a minority, normal or subnormal
    let num_minority_values = num_subnormals.min(data.as_ref().len() - num_subnormals);

    // Figure out how many of them are fed to each accumulator on average
    // (assuming normal/subnormal values are either randomly or evenly
    // distributed between accumulators, which should be the case)
    let mut values_per_acc = num_minority_values;
    if !I::KIND.is_reused() {
        values_per_acc /= ilp;
    };

    // Figure out how many values there are in a single input substream feeding
    // a single accumulator. That's the part that needs to be varied enough.
    let values_per_substream = values_per_acc / substreams_per_accumulator;

    // If that is too small, request that inputs are regenerated each run
    values_per_substream < 1000
}

/// Initialize accumulators with a positive value that is suitable as the
/// starting point of an additive random walk of ~unity step.
///
/// The chosen initial value takes a half-way compromise between two objectives:
///
/// - When many positive values are subtracted, it should not get close to zero
/// - When many positive values are added, it should not saturate to a maximum
#[inline]
pub fn additive_accumulators<T: FloatLike, const ILP: usize>(rng: &mut impl Rng) -> [T; ILP] {
    let normal_sampler = T::normal_sampler();
    std::array::from_fn::<_, ILP, _>(|_| {
        T::splat(2.0f32.powi((T::MANTISSA_DIGITS / 2) as i32)) * normal_sampler(rng)
    })
}

/// Initialize accumulators with a positive value close to 1.0
///
/// This is the right default for most benchmarks, except those that perform an
/// additive random walk.
#[inline]
pub fn normal_accumulators<T: FloatLike, const ILP: usize>(rng: &mut impl Rng) -> [T; ILP] {
    let normal_sampler = T::normal_sampler();
    std::array::from_fn::<_, ILP, _>(|_| normal_sampler(rng))
}

/// Benchmark skeleton that processes the full input homogeneously
#[inline]
pub fn integrate_full<
    T: FloatLike,
    I: Inputs<Element = T>,
    const ILP: usize,
    const HIDE_INPUTS: bool,
>(
    accumulators: &mut [T; ILP],
    mut hide_accumulators: impl FnMut(&mut [T; ILP]),
    inputs: &mut I,
    mut iter: impl Copy + FnMut(T, T) -> T,
) {
    let inputs_slice = inputs.as_ref();
    if I::KIND.is_reused() {
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
                inputs.hide_inplace();
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
pub fn integrate_halves<T: FloatLike, I: Inputs<Element = T>, const ILP: usize>(
    accumulators: &mut [T; ILP],
    inputs: &I,
    mut low_iter: impl Copy + FnMut(T, T) -> T,
    mut high_iter: impl Copy + FnMut(T, T) -> T,
) {
    if I::KIND.is_reused() {
        // Otherwise, we just do the same as usual, but with input reuse
        let inputs_slice = inputs.as_ref();
        let (low_inputs, high_inputs) = inputs_slice.split_at(inputs_slice.len() / 2);
        assert_eq!(low_inputs.len(), high_inputs.len());
        for (&low_elem, &high_elem) in low_inputs.iter().zip(high_inputs) {
            for acc in accumulators.iter_mut() {
                *acc = low_iter(*acc, low_elem);
                *acc = high_iter(*acc, high_elem);
            }
            hide_accumulators::<_, ILP, true>(accumulators);
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
            hide_accumulators::<_, ILP, true>(accumulators);
        }
        for (&low_elem, acc) in low_remainder.iter().zip(accumulators.iter_mut()) {
            *acc = low_iter(*acc, low_elem);
        }
        for (&high_elem, acc) in high_remainder.iter().zip(accumulators.iter_mut()) {
            *acc = high_iter(*acc, high_elem);
        }
    }
}

/// Minimal optimization barrier to avoid accumulator autovectorization
///
/// If we accumulate data naively, LLVM can figure out that we're aggregating
/// arrays of inputs into identically sized arrays of accumulators, and
/// helpfully autovectorize the accumulation. Sadly, we do not want this
/// optimization here because it gets in the way of us separately studying the
/// hardware overhead of scalar and SIMD operations.
///
/// To prevent this, we send accumulators through opaque inline ASM via
/// [`pessimize::hide()`] after ~each accumulation iteration. This means that in
/// order to autovectorize, LLVM must now group data in wide vectors, then split
/// it back into narrow vctors, on each iteration. For any simple computation,
/// that's a strong enough deterrent to foil its autovectorization plans.
///
/// Unfortunately, these optimization barriers also negatively affects code
/// generation. For example, they can cause unnecessary register-register copies
/// and thus increase CPU register pressure above the optimum for a certain
/// numerical computation. And they can cause unrelated data that is resident in
/// CPU registers to spill to memory and be loaded back into registers. To
/// minimize these side-effects, you want as few optimization barriers as
/// possible, targeting as little data as possible.
///
/// In MINIMAL mode, this function function tries to apply the minimum viable
/// accumulator-hiding barrier for autovectorization prevention in easy cases.
/// It does not work for all computations, but when it works, it can lead to
/// lower overhead than putting a barrier on all accumulators. If it doesn't
/// work, just set MINIMAL to false and we'll hide all the accumulators if there is
/// any risk of autovectorization.
///
/// To improve codegen further, you can also abandon the convenience of `-C
/// target-cpu=native` and instead separately benchmark each type in a dedicated
/// configuration that does not enable unnecessary wide vector instruction sets.
/// For example, on  x86_64, you would want `RUSTFLAGS='-C
/// target_feature=+avx,+fma'` for f32x08 and f64x04, and `RUSTFLAGS=''` for SSE
/// and scalar types (it's not legal to disable SSE on x86_64).
#[inline]
pub fn hide_accumulators<T: FloatLike, const ILP: usize, const MINIMAL: bool>(
    accumulators: &mut [T; ILP],
) {
    // No need for pessimize::hide() optimization barriers if this accumulation
    // cannot be autovectorized because e.g. we are operating at the maximum
    // hardware-supported SIMD vector width.
    let Some(min_vector_ilp) = T::MIN_VECTORIZABLE_ILP else {
        return;
    };

    // Otherwise apply pessimize::hide() to a set of accumulators which is large
    // enough that there aren't enough remaining non-hidden accumulators left
    // for the compiler to perform reasonable autovectorization.
    let max_elided_barriers = if MINIMAL { min_vector_ilp.get() - 1 } else { 0 };
    let min_hidden_accs = ILP.saturating_sub(max_elided_barriers);
    for acc in accumulators.iter_mut().take(min_hidden_accs) {
        let old_acc = *acc;
        let new_acc = pessimize::hide::<T>(old_acc);
        *acc = new_acc;
    }
}

/// Single-accumulator autovectorization barrier
///
/// Use in situations where a coarse-grained [`hide_accumulators()`] barrier
/// that applies to the full set of accumulators is not enough to reliably
/// prevent autovectorization. This typically happens inside of sub-expressions
/// of a computation that LLVM considers complex enough to justify internal
/// autovectorization.
#[inline]
pub fn hide_single_accumulator<T: FloatLike>(accumulator: T) -> T {
    if T::MIN_VECTORIZABLE_ILP.is_some() {
        pessimize::hide::<T>(accumulator)
    } else {
        accumulator
    }
}
