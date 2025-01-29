//! Benchmark inner loops and associated types

use crate::{
    floats::{self, FloatLike},
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
    fn make_benchmark<Storage: InputsMut, const ILP: usize>(
        input_storage: Storage,
    ) -> impl Benchmark<Float = Storage::Element>;
}
//
/// Microbenchmark of a certain [`Operation`]
pub trait Benchmark {
    /// Floating-point type that is being exercised
    type Float: FloatLike;

    /// Number of hardware operations under study that this benchmark will
    /// execute when processing the previously specified input dataset
    ///
    /// Most benchmarks consume one input element per computed operation, in
    /// which case this will be [`accumulated_len(&input_storage,
    /// ilp)`](accumulated_len), with `input_storage` pointing to the previously
    /// specified input storage and `ilp` set to this benchmark's degree of
    /// accumulation instruction-level parallelism.
    ///
    /// However, beware of benchmarks like `fma_full` which consume multiple
    /// input elements per computed operation.
    fn num_operations(&self) -> usize;

    /// Specify the desired share of subnormal numbers in the input dataset.
    ///
    /// # Panics
    ///
    /// Panics if the specified `num_subnormals` is greater than the length of
    /// the underlying input storage.
    fn setup_inputs(&mut self, num_subnormals: usize);

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
    /// then the following alternatives may be considered:
    ///
    /// - Keep the same values, but randomly shuffle them. This is good enough
    ///   for relatively large sets of random values, like in-RAM input
    ///   datasets, where there is no meaningful difference between a randomly
    ///   reordered dataset and a fully regenerated one.
    /// - Keep the same values, just pass them through [`pessimize::hide()`] so
    ///   the compiler doesn't know that they are the same. This should be a
    ///   last-resort option, as it maximizes measurement bias.
    ///
    /// You will normally want to initialize the accumulators using
    /// [`narrow_accumulators()`] in this function, to avoid the precision
    /// issues that arise when inputs and accumulators do not have the same
    /// order of magnitude. But in a benchmarks of `MAX` and similar primitives
    /// that have no accumulation error you can go for the maximally general
    /// [`normal_accumulators()`].
    ///
    /// The input storage of the resulting [`BenchmarkRun`] should be derived
    /// from that of this benchmark through
    /// [`clone_or_reborrow()`](InputStorage::clone_or_reborrow).
    ///
    /// Implementations should be marked `#[inline]` to give the compiler a more
    /// complete picture of the accumulator data flow.
    fn start_run(&mut self, rng: &mut impl Rng) -> Self::Run<'_>;

    /// State of an ongoing execution of this benchmark
    type Run<'run>: BenchmarkRun<Float = Self::Float>
    where
        Self: 'run;
}
//
/// Ongoing execution of a certain [`Benchmark`]
pub trait BenchmarkRun {
    /// Floating-point type that is being exercised
    type Float: FloatLike;

    /// Access the internal input dataset
    ///
    /// This API is only meant for use by the unit test procedures defined in
    /// `test_utils` and should not be used for other purposes.
    fn inputs(&self) -> &[Self::Float];

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
    /// called repeatedly as part of the timed benchmark run.
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

/// Total number of inputs that get aggregated into a benchmark's accumulators
///
/// This accounts for the reuse of small in-register input datasets across all
/// benchmark accumulators.
pub fn accumulated_len<I: Inputs>(inputs: &I, ilp: usize) -> usize {
    let mut result = inputs.as_ref().len();
    if I::KIND.is_reused() {
        result *= ilp;
    }
    result
}

/// Initialize accumulators with a random positive value in range [0.5; 2[
#[inline]
pub fn narrow_accumulators<T: FloatLike, const ILP: usize>(rng: &mut impl Rng) -> [T; ILP] {
    let narrow = floats::narrow_sampler();
    std::array::from_fn::<_, ILP, _>(|_| narrow(rng))
}

/// Initialize accumulators with a random normal positive value
#[inline]
pub fn normal_accumulators<T: FloatLike, const ILP: usize>(rng: &mut impl Rng) -> [T; ILP] {
    let normal = floats::normal_sampler();
    std::array::from_fn::<_, ILP, _>(|_| normal(rng))
}

/// Benchmark skeleton for processing a dataset element-wise with N accumulators
///
/// `hide_accumulators` should usually point to an instance of the
/// [`hide_accumulators()`] function defined in this module. The only reason why
/// we don't make it just a bool generic parameter is that sets of multiple
/// generic boolean parameters are hard to read on the caller side.
#[inline]
pub fn integrate<
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

/// Benchmark skeleton for processing a dataset pair-wise with N accumulators
///
/// This works a lot like [`integrate()`] except the dataset is first split in
/// two halves, then we take pairs of elements from each half and integrate the
/// resulting input pair into the accumulator in a single transaction.
///
/// This is useful for operations like `fma_addend` which treat inputs
/// inhomogeneously and for operations like `fma_full` which require two inputs
/// per elementary accumulation transaction.
#[inline]
pub fn integrate_pairs<T: FloatLike, I: Inputs<Element = T>, const ILP: usize>(
    accumulators: &mut [T; ILP],
    mut hide_accumulators: impl FnMut(&mut [T; ILP]),
    inputs: &I,
    mut iter: impl Copy + FnMut(T, [T; 2]) -> T,
) {
    let inputs = inputs.as_ref();
    let inputs_len = inputs.len();
    assert_eq!(inputs_len % 2, 0);
    let (left, right) = inputs.split_at(inputs_len / 2);
    if I::KIND.is_reused() {
        for (&left_elem, &right_elem) in left.iter().zip(right) {
            for acc in accumulators.iter_mut() {
                *acc = iter(*acc, [left_elem, right_elem]);
            }
            hide_accumulators(accumulators);
        }
    } else {
        let left_chunks = left.chunks_exact(ILP);
        let right_chunks = right.chunks_exact(ILP);
        let left_remainder = left_chunks.remainder();
        let right_remainder = right_chunks.remainder();
        for (left_chunk, right_chunk) in left_chunks.zip(right_chunks) {
            for ((&left_elem, &right_elem), acc) in left_chunk
                .iter()
                .zip(right_chunk)
                .zip(accumulators.iter_mut())
            {
                *acc = iter(*acc, [left_elem, right_elem]);
            }
            hide_accumulators(accumulators);
        }
        for ((&left_elem, &right_elem), acc) in left_remainder
            .iter()
            .zip(right_remainder)
            .zip(accumulators.iter_mut())
        {
            *acc = iter(*acc, [left_elem, right_elem]);
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
/// work, just set MINIMAL to false and we'll hide all the accumulators if there
/// is any risk of autovectorization.
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

/// Test utilities that are used by other crates in the workspace
#[cfg(feature = "unstable_test")]
pub mod test_utils {
    use super::*;
    use crate::{
        arch::MIN_FLOAT_REGISTERS,
        floats::test_utils::FloatLikeExt,
        inputs::{
            generators::test_utils::num_subnormals,
            test_utils::{f32_array, f32_vec},
        },
        tests::assert_panics,
    };
    use num_traits::NumCast;
    use proptest::prelude::*;
    use std::panic::AssertUnwindSafe;

    /// Number of benchmark iterations performed by operation tests
    const NUM_TEST_ITERATIONS: usize = 100;

    /// Truth that a certain operation requires the benchmark's accumulators to
    /// have a "narrow" magnitude (in range `[1/2; 2]`) at the start of a
    /// benchmark iteration.
    ///
    /// This is the type of the `$needs_narrow_accs` argument to the
    /// `test_xyz_operation!()` macros.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum NeedsNarrowAcc {
        /// Accumulators should always have a narrow magnitude at the start of a
        /// benchmark iteration.
        ///
        /// This is appropriate for "additive" benchmarks where integrating a
        /// subnormal input into an accumulator will not destroy the original
        /// accumulator magnitude information.
        Always,

        /// Accumulators should initially have a narrow magnitude as in
        /// `Always`, but are allowed to have a non-narrow magnitude at the
        /// **end** of a benchmark iteration (and thus at the start of the next
        /// iteration) if the first integrated input is subnormal.
        ///
        /// This is appropriate for "multiplicative" benchmarks where
        /// integrating a subnormal input into an accumulator will destroy the
        /// original accumulator magnitude information.
        ///
        /// For operations that work on pairs of inputs, the first input pair is
        /// only considered to be a subnormal input if both elements from the
        /// input pair are subnormal.
        FirstNormal,

        /// Accumulators never needs to have a narrow magnitude.
        ///
        /// This is appropriate for "comparative" benchmarks where the initial
        /// accumulator magnitude does not meaningfully affect the benchmark's
        /// behavior, as long as the accumulator remains a normal number.
        Never,
    }

    /// Generate unit tests for a certain [`Operation`] where each accumulation
    /// step ingests one input element
    #[macro_export]
    macro_rules! test_scalar_operation {
        ($op:ty, $needs_narrow_acc:expr) => {
            $crate::test_scalar_operation!($op, $needs_narrow_acc, 0.0f32);
        };
        ($op:ty, $needs_narrow_acc:expr, $non_narrow_tolerance_per_iter:expr) => {
            mod operation {
                use super::*;
                use $crate::{
                    inputs::test_utils::f32_array,
                    operations::test_utils::f32_array_and_num_nubnormals,
                };
                $crate::_impl_operation_tests!(
                    $op,
                    $needs_narrow_acc,
                    $non_narrow_tolerance_per_iter,
                    false,
                    1,
                );

                proptest! {
                    #[test]
                    fn num_benchmark_operations_1reg_ilp1(regs in f32_array::<1>()) {
                        test_num_benchmark_operations::<1>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_1reg_ilp2(regs in f32_array::<1>()) {
                        test_num_benchmark_operations::<2>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_1reg_ilp3(regs in f32_array::<1>()) {
                        test_num_benchmark_operations::<3>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_2regs_ilp1(regs in f32_array::<2>()) {
                        test_num_benchmark_operations::<1>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_2regs_ilp2(regs in f32_array::<2>()) {
                        test_num_benchmark_operations::<2>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_2regs_ilp3(regs in f32_array::<2>()) {
                        test_num_benchmark_operations::<3>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_3regs_ilp1(regs in f32_array::<3>()) {
                        test_num_benchmark_operations::<1>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_3regs_ilp2(regs in f32_array::<3>()) {
                        test_num_benchmark_operations::<2>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_3regs_ilp3(regs in f32_array::<3>()) {
                        test_num_benchmark_operations::<3>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_4regs_ilp1(regs in f32_array::<4>()) {
                        test_num_benchmark_operations::<1>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_4regs_ilp2(regs in f32_array::<4>()) {
                        test_num_benchmark_operations::<2>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_4regs_ilp3(regs in f32_array::<4>()) {
                        test_num_benchmark_operations::<3>(regs)?;
                    }

                    #[test]
                    fn benchmark_run_1reg_ilp1((regs, num_subnormals) in f32_array_and_num_nubnormals::<1>()) {
                        test_benchmark_run::<_, 1>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_1reg_ilp2((regs, num_subnormals) in f32_array_and_num_nubnormals::<1>()) {
                        test_benchmark_run::<_, 2>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_1reg_ilp3((regs, num_subnormals) in f32_array_and_num_nubnormals::<1>()) {
                        test_benchmark_run::<_, 3>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_2regs_ilp1((regs, num_subnormals) in f32_array_and_num_nubnormals::<2>()) {
                        test_benchmark_run::<_, 1>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_2regs_ilp2((regs, num_subnormals) in f32_array_and_num_nubnormals::<2>()) {
                        test_benchmark_run::<_, 2>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_2regs_ilp3((regs, num_subnormals) in f32_array_and_num_nubnormals::<2>()) {
                        test_benchmark_run::<_, 3>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_3regs_ilp1((regs, num_subnormals) in f32_array_and_num_nubnormals::<3>()) {
                        test_benchmark_run::<_, 1>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_3regs_ilp2((regs, num_subnormals) in f32_array_and_num_nubnormals::<3>()) {
                        test_benchmark_run::<_, 2>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_3regs_ilp3((regs, num_subnormals) in f32_array_and_num_nubnormals::<3>()) {
                        test_benchmark_run::<_, 3>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_4regs_ilp1((regs, num_subnormals) in f32_array_and_num_nubnormals::<4>()) {
                        test_benchmark_run::<_, 1>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_4regs_ilp2((regs, num_subnormals) in f32_array_and_num_nubnormals::<4>()) {
                        test_benchmark_run::<_, 2>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_4regs_ilp3((regs, num_subnormals) in f32_array_and_num_nubnormals::<4>()) {
                        test_benchmark_run::<_, 3>(regs, num_subnormals)?;
                    }
                }
            }
        };
    }

    /// Generate unit tests for a certain [`Operation`] where each accumulation
    /// step ingests two input element
    #[macro_export]
    macro_rules! test_pairwise_operation {
        ($op:ty, $needs_narrow_acc:expr, $inputs_per_op:literal) => {
            $crate::test_pairwise_operation!($op, $needs_narrow_acc, $inputs_per_op, 0.0f32);
        };
        ($op:ty, $needs_narrow_acc:expr, $inputs_per_op:literal, $non_narrow_tolerance_per_iter:expr) => {
            mod operation {
                use super::*;
                use $crate::{
                    inputs::test_utils::f32_array,
                    operations::test_utils::f32_array_and_num_nubnormals,
                };
                $crate::_impl_operation_tests!(
                    $op,
                    $needs_narrow_acc,
                    $non_narrow_tolerance_per_iter,
                    true,
                    $inputs_per_op,
                );

                proptest! {
                    #[test]
                    fn num_benchmark_operations_2regs_ilp1(regs in f32_array::<2>()) {
                        test_num_benchmark_operations::<1>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_2regs_ilp2(regs in f32_array::<2>()) {
                        test_num_benchmark_operations::<2>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_2regs_ilp3(regs in f32_array::<2>()) {
                        test_num_benchmark_operations::<3>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_4regs_ilp1(regs in f32_array::<4>()) {
                        test_num_benchmark_operations::<1>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_4regs_ilp2(regs in f32_array::<4>()) {
                        test_num_benchmark_operations::<2>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_4regs_ilp3(regs in f32_array::<4>()) {
                        test_num_benchmark_operations::<3>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_6regs_ilp1(regs in f32_array::<6>()) {
                        test_num_benchmark_operations::<1>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_6regs_ilp2(regs in f32_array::<6>()) {
                        test_num_benchmark_operations::<2>(regs)?;
                    }

                    #[test]
                    fn num_benchmark_operations_6regs_ilp3(regs in f32_array::<6>()) {
                        test_num_benchmark_operations::<3>(regs)?;
                    }

                    #[test]
                    fn benchmark_run_2regs_ilp1((regs, num_subnormals) in f32_array_and_num_nubnormals::<2>()) {
                        test_benchmark_run::<_, 1>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_2regs_ilp2((regs, num_subnormals) in f32_array_and_num_nubnormals::<2>()) {
                        test_benchmark_run::<_, 2>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_2regs_ilp3((regs, num_subnormals) in f32_array_and_num_nubnormals::<2>()) {
                        test_benchmark_run::<_, 3>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_4regs_ilp1((regs, num_subnormals) in f32_array_and_num_nubnormals::<4>()) {
                        test_benchmark_run::<_, 1>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_4regs_ilp2((regs, num_subnormals) in f32_array_and_num_nubnormals::<4>()) {
                        test_benchmark_run::<_, 2>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_4regs_ilp3((regs, num_subnormals) in f32_array_and_num_nubnormals::<4>()) {
                        test_benchmark_run::<_, 3>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_6regs_ilp1((regs, num_subnormals) in f32_array_and_num_nubnormals::<6>()) {
                        test_benchmark_run::<_, 1>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_6regs_ilp2((regs, num_subnormals) in f32_array_and_num_nubnormals::<6>()) {
                        test_benchmark_run::<_, 2>(regs, num_subnormals)?;
                    }

                    #[test]
                    fn benchmark_run_6regs_ilp3((regs, num_subnormals) in f32_array_and_num_nubnormals::<6>()) {
                        test_benchmark_run::<_, 3>(regs, num_subnormals)?;
                    }
                }
            }
        };
    }

    /// Common parts of [`test_unary_operation`] and [`test_binary_operation`]
    ///
    /// Must be instantiated in a dedicated module to avoid name collisions
    #[doc(hidden)]
    #[macro_export]
    macro_rules! _impl_operation_tests {
        ($op:ty, $needs_narrow_acc:expr, $non_narrow_tolerance_per_iter:expr, $pairwise:literal, $inputs_per_op:literal,) => {
            use super::*;
            use $crate::{
                floats::test_utils::FloatLikeExt,
                inputs::test_utils::f32_vec,
                operations::test_utils::{
                    self as operation_utils, f32_vec_and_num_subnormals,
                },
                proptest::prelude::*,
            };

            #[test]
            fn metadata() {
                operation_utils::test_operation_metadata::<$op>();
            }

            fn test_num_benchmark_operations<const ILP: usize>(
                input_storage: impl InputsMut,
            ) -> Result<(), TestCaseError> {
                operation_utils::test_num_benchmark_operations::<$op, _, { ILP }>(
                    input_storage,
                    $pairwise,
                    $inputs_per_op,
                )
            }

            proptest! {
                #[test]
                fn num_benchmark_operations_mem_ilp1(mut mem in f32_vec($pairwise)) {
                    test_num_benchmark_operations::<1>(&mut mem[..])?;
                }

                #[test]
                fn num_benchmark_operations_mem_ilp2(mut mem in f32_vec($pairwise)) {
                    test_num_benchmark_operations::<2>(&mut mem[..])?;
                }

                #[test]
                fn num_benchmark_operations_mem_ilp3(mut mem in f32_vec($pairwise)) {
                    test_num_benchmark_operations::<3>(&mut mem[..])?;
                }
            }

            fn test_benchmark_run<Storage: InputsMut, const ILP: usize>(
                input_storage: Storage,
                num_subnormals: usize,
            ) -> Result<(), TestCaseError> where Storage::Element: FloatLikeExt {
                operation_utils::test_benchmark_run::<$op, _, { ILP }>(
                    input_storage,
                    $pairwise,
                    num_subnormals,
                    $needs_narrow_acc,
                    $non_narrow_tolerance_per_iter,
                )
            }

            proptest! {
                #[test]
                fn benchmark_run_mem_ilp1((mut mem, num_subnormals) in f32_vec_and_num_subnormals($pairwise)) {
                    test_benchmark_run::<_, 1>(&mut mem[..], num_subnormals)?;
                }

                #[test]
                fn benchmark_run_mem_ilp2((mut mem, num_subnormals) in f32_vec_and_num_subnormals($pairwise)) {
                    test_benchmark_run::<_, 2>(&mut mem[..], num_subnormals)?;
                }

                #[test]
                fn benchmark_run_mem_ilp3((mut mem, num_subnormals) in f32_vec_and_num_subnormals($pairwise)) {
                    test_benchmark_run::<_, 3>(&mut mem[..], num_subnormals)?;
                }
            }
        };
    }

    /// Generate a array of f32s and a matching num_subnormals configuration
    #[doc(hidden)]
    pub fn f32_array_and_num_nubnormals<const SIZE: usize>(
    ) -> impl Strategy<Value = ([f32; SIZE], usize)> {
        (f32_array(), num_subnormals(SIZE))
    }

    /// Generate a Vec of f32s and a matching num_subnormals configuration
    #[doc(hidden)]
    pub fn f32_vec_and_num_subnormals(pairwise: bool) -> impl Strategy<Value = (Vec<f32>, usize)> {
        f32_vec(pairwise).prop_flat_map(|v| {
            let v_len = v.len();
            (Just(v), num_subnormals(v_len))
        })
    }

    /// Check that operation metadata makes basic sense
    #[doc(hidden)]
    pub fn test_operation_metadata<Op: Operation>() {
        assert!(!Op::NAME.is_empty());
        assert!(Op::AUX_REGISTERS_MEMOP < MIN_FLOAT_REGISTERS);
        assert!(Op::aux_registers_regop(2) < MIN_FLOAT_REGISTERS);
    }

    /// Instantiate a benchmark and check its advertised operation count
    #[doc(hidden)]
    pub fn test_num_benchmark_operations<Op: Operation, Storage: InputsMut, const ILP: usize>(
        input_storage: Storage,
        pairwise: bool,
        inputs_per_op: usize,
    ) -> Result<(), TestCaseError> {
        // Determine and check effective input length
        let accumulated_len = accumulated_len(&input_storage, ILP);

        // Instantiate benchmark, checking for invalid input length
        let input_len_granularity = 1 + (pairwise as usize);
        let benchmark = if input_storage.as_ref().len() % input_len_granularity != 0 {
            return assert_panics(AssertUnwindSafe(|| {
                Op::make_benchmark::<_, ILP>(input_storage)
            }));
        } else {
            Op::make_benchmark::<_, ILP>(input_storage)
        };

        // Check that the number of operation matches expectations
        prop_assert_eq!(benchmark.num_operations(), accumulated_len / inputs_per_op);
        Ok(())
    }

    /// Simulate a benchmark run
    #[doc(hidden)]
    pub fn test_benchmark_run<Op: Operation, Storage: InputsMut, const ILP: usize>(
        input_storage: Storage,
        pairwise: bool,
        num_subnormals: usize,
        needs_narrow_acc: NeedsNarrowAcc,
        non_narrow_tolerance_per_iter: f32,
    ) -> Result<(), TestCaseError>
    where
        Storage::Element: FloatLikeExt,
    {
        // Collect input_storage metadata while we can
        let input_slice = input_storage.as_ref();
        let input_len = input_slice.len();
        let input_is_reused = Storage::KIND.is_reused();

        // Set up a benchmark...
        let mut benchmark = if pairwise && input_len % 2 != 0 {
            return assert_panics(AssertUnwindSafe(|| {
                Op::make_benchmark::<_, ILP>(input_storage)
            }));
        } else {
            Op::make_benchmark::<_, ILP>(input_storage)
        };

        // ...then specify the number of operations...
        let initial_num_ops = benchmark.num_operations();
        if num_subnormals > input_len {
            return assert_panics(AssertUnwindSafe(|| benchmark.setup_inputs(num_subnormals)));
        } else {
            benchmark.setup_inputs(num_subnormals);
        }
        prop_assert_eq!(benchmark.num_operations(), initial_num_ops);

        // ...and finally set up a benchmark run
        {
            let mut run = benchmark.start_run(&mut rand::thread_rng());

            // Check generated input values
            let mut actual_subnormals = 0;
            let initial_inputs = run.inputs().to_owned();
            assert_eq!(initial_inputs.len(), input_len);
            for &input in &initial_inputs {
                if input.is_subnormal() {
                    actual_subnormals += 1;
                } else {
                    prop_assert!(input.is_normal() || input == Storage::Element::splat(0.0));
                }
            }
            prop_assert_eq!(actual_subnormals, num_subnormals);

            // Determine if the first input that is going to be fed into each
            // accumulator is normal.
            let input_is_normal =
                |idx: usize| initial_inputs.get(idx).is_some_and(|x| x.is_normal());
            let mut first_input_normal = if input_is_reused {
                [input_is_normal(0); ILP]
            } else {
                std::array::from_fn(input_is_normal)
            };
            if pairwise {
                debug_assert_eq!(input_len % 2, 0);
                let first_right_input = input_len / 2;
                if input_is_reused {
                    first_input_normal = std::array::from_fn(|acc_idx| {
                        first_input_normal[acc_idx] && input_is_normal(first_right_input)
                    });
                } else {
                    first_input_normal = std::array::from_fn(|acc_idx| {
                        first_input_normal[acc_idx] && input_is_normal(first_right_input + acc_idx)
                    });
                }
            }

            // Check for initial accumulator values
            fn check_accs<R: BenchmarkRun>(
                run: &R,
                first_input_normal: &[bool],
                needs_narrow_acc: NeedsNarrowAcc,
                tolerance: f32,
                error_context: impl Fn() -> String,
            ) -> Result<(), TestCaseError>
            where
                R::Float: FloatLikeExt,
            {
                debug_assert_eq!(first_input_normal.len(), run.accumulators().len());
                for (acc_idx, (acc, &first_input_normal)) in run
                    .accumulators()
                    .iter()
                    .zip(first_input_normal)
                    .enumerate()
                {
                    let error_context = || {
                        format!(
                            "{}\n* Investigating accumulator #{acc_idx} ({acc:?})",
                            error_context()
                        )
                    };
                    let needs_narrow_acc = match needs_narrow_acc {
                        NeedsNarrowAcc::Always => true,
                        NeedsNarrowAcc::FirstNormal => first_input_normal,
                        NeedsNarrowAcc::Never => false,
                    };
                    if needs_narrow_acc {
                        for scalar in acc.as_scalars() {
                            let scalar_from_f32 = |x: f32| {
                                <<R::Float as FloatLikeExt>::Scalar as NumCast>::from(x).unwrap()
                            };
                            let error = || {
                                format!(
                                    "{}\n* Accumulator value {scalar:?} escaped expected \"narrow\" range [1/2; 2] by more than tolerance {tolerance}",
                                    error_context()
                                )
                            };
                            prop_assert!(
                                *scalar >= scalar_from_f32(0.5 - tolerance),
                                "{}",
                                error()
                            );
                            prop_assert!(
                                *scalar <= scalar_from_f32(2.0 + tolerance),
                                "{}",
                                error()
                            );
                        }
                    } else {
                        prop_assert!(acc.is_normal());
                    }
                }
                Ok(())
            }

            // Start building an error report
            let initial_accs = run.accumulators().to_owned();
            let error_context = || format!("\n* Starting from accumulators ({initial_accs:?})");

            // Before the first iteration, if the benchmark wants a narrow
            // accumulator magnitude, the accumulator should have a narrow
            // magnitude. The "FirstNormal" tolerance only applies after the
            // inputs have been integrated into the accumulator.
            let needs_narrow_initial_acc = match needs_narrow_acc {
                NeedsNarrowAcc::Always | NeedsNarrowAcc::FirstNormal => NeedsNarrowAcc::Always,
                NeedsNarrowAcc::Never => NeedsNarrowAcc::Never,
            };
            check_accs(
                &run,
                &first_input_normal,
                needs_narrow_initial_acc,
                0.0,
                error_context,
            )?;

            // Perform a number of benchmark iterations
            let mut accumulator_values = Vec::new();
            for i in 1..=NUM_TEST_ITERATIONS {
                // Integrate the inputs
                run.integrate_inputs();

                // Check that run invariants are preserved over iterations
                prop_assert_eq!(run.inputs(), &initial_inputs);
                check_accs(
                    &run,
                    &first_input_normal,
                    needs_narrow_acc,
                    non_narrow_tolerance_per_iter * i as f32,
                    || {
                        let mut result = format!(
                            "{}\n* After {i} iteration(s) of integrating input(s) {initial_inputs:?}",
                            error_context()
                        );
                        if !accumulator_values.is_empty() {
                            result.push_str(", which got us through accumulator values...");
                        }
                        for acc in accumulator_values.iter() {
                            result.push_str(&format!("\n  - {acc:?}"));
                        }
                        result
                    },
                )?;
                accumulator_values.push(run.accumulators().to_owned());
            }
        }

        // Check that benchmark invariants are preserved over iterations
        prop_assert_eq!(benchmark.num_operations(), initial_num_ops);
        Ok(())
    }
}
