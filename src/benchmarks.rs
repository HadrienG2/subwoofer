//! Bits that are shared by multiple benchmarks
//!
//! All benchmarks use one of the `iter_` inner loop skeletons, except for
//! `fma_full` which implements its own variation of [`iter_halves()`] because
//! FMA is the only benchmarked operation that takes two input data operands.

use crate::types::{FloatLike, FloatSequence};

/// Maximum granularity of subnormal occurence probabilities
///
/// Higher is more precise, but the benchmark execution time in the default
/// configuration is multipled accordingly.
pub const MAX_SUBNORMAL_CONFIGURATIONS: usize = const {
    if cfg!(feature = "subnormal_freq_resolution_1in128") {
        128 // 0.78125% granularity
    } else if cfg!(feature = "subnormal_freq_resolution_1in64") {
        64 // 1.5625% granularity
    } else if cfg!(feature = "subnormal_freq_resolution_1in32") {
        32 // 3.125% granularity
    } else if cfg!(feature = "subnormal_freq_resolution_1in16") {
        16 // 6.25% granularity
    } else if cfg!(feature = "subnormal_freq_resolution_1in8") {
        8 // 12.5% granularity
    } else {
        4 // 25% granularity
    }
};

/// Benchmark skeleton that processes the full input homogeneously
#[inline]
pub fn iter_full<T: FloatLike, Inputs: FloatSequence<T>, const ILP: usize>(
    mut accumulators: [T; ILP],
    inputs: Inputs,
    mut iter: impl Copy + FnMut(T, T) -> T,
) -> ([T; ILP], Inputs) {
    let inputs_slice = inputs.as_ref();
    if Inputs::IS_REUSED {
        for &elem in inputs_slice {
            for acc in accumulators.iter_mut() {
                *acc = iter(*acc, elem);
            }
            accumulators = hide_accumulators(accumulators);
        }
    } else {
        let chunks = inputs_slice.chunks_exact(ILP);
        let remainder = chunks.remainder();
        for chunk in chunks {
            for (&elem, acc) in chunk.iter().zip(accumulators.iter_mut()) {
                *acc = iter(*acc, elem);
            }
            accumulators = hide_accumulators(accumulators);
        }
        for (&elem, acc) in remainder.iter().zip(accumulators.iter_mut()) {
            *acc = iter(*acc, elem);
        }
    }
    (accumulators, inputs)
}

/// Benchmark skeleton that treats each half of the input differently
#[inline]
pub fn iter_halves<
    T: FloatLike,
    Inputs: FloatSequence<T>,
    const ILP: usize,
    const HIDE_INPUTS: bool,
>(
    mut accumulators: [T; ILP],
    mut inputs: Inputs,
    mut low_iter: impl Copy + FnMut(T, T) -> T,
    mut high_iter: impl Copy + FnMut(T, T) -> T,
) -> ([T; ILP], Inputs) {
    if Inputs::IS_REUSED {
        if HIDE_INPUTS {
            // When we need to hide inputs, we flip the order of iteration over
            // inputs and accumulators so that each set of inputs is fully
            // processed before we switch accumulators.
            //
            // This ensures that the optimization barrier is maximally allowed
            // to reuse the same registers for the original and hidden inputs,
            // at the expense of making it harder for the CPU to extract ILP if
            // the backend doesn't reorder instructions because the CPU frontend
            // must now dive through N mentions of the same accumulator before
            // reaching mentions of another accumulator (no unroll & jam).
            for acc in accumulators.iter_mut() {
                let inputs_slice = inputs.as_ref();
                let (low_inputs, high_inputs) = inputs_slice.split_at(inputs_slice.len() / 2);
                assert_eq!(low_inputs.len(), high_inputs.len());
                for (&low_elem, &high_elem) in low_inputs.iter().zip(high_inputs) {
                    *acc = low_iter(*acc, low_elem);
                    *acc = high_iter(*acc, high_elem);
                }
                // Autovectorization barrier must target one accumulator at a
                // time here due to the input/accumulator loop transpose.
                *acc = pessimize::hide::<T>(*acc);
                inputs = <Inputs as FloatSequence<T>>::hide(inputs);
            }
        } else {
            // Otherwise, we just do the same as usual, but with input reuse
            let inputs_slice = inputs.as_ref();
            let (low_inputs, high_inputs) = inputs_slice.split_at(inputs_slice.len() / 2);
            assert_eq!(low_inputs.len(), high_inputs.len());
            for (&low_elem, &high_elem) in low_inputs.iter().zip(high_inputs) {
                for acc in accumulators.iter_mut() {
                    *acc = low_iter(*acc, low_elem);
                    *acc = high_iter(*acc, high_elem);
                }
                accumulators = hide_accumulators(accumulators);
            }
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
            accumulators = hide_accumulators(accumulators);
        }
        for (&low_elem, acc) in low_remainder.iter().zip(accumulators.iter_mut()) {
            *acc = low_iter(*acc, low_elem);
        }
        for (&high_elem, acc) in high_remainder.iter().zip(accumulators.iter_mut()) {
            *acc = high_iter(*acc, high_elem);
        }
    }
    (accumulators, inputs)
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
/// re-splitting it into narrow vectores is higher than the performance gain of
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
/// x86_64, that would be `-C target_feature=+avx` for f32x08 and f64x04, `-C
/// target_feature=+avx512f` for f32x16 and f64x08, and nothing for all other
/// types (it is not legal to disable SSE for better scalar codegen on x86_64).
#[inline]
pub fn hide_accumulators<T: FloatLike, const ILP: usize>(mut accumulators: [T; ILP]) -> [T; ILP] {
    // No need for pessimize::hide() optimization barriers if this accumulation
    // cannot be autovectorized because e.g. we are operating at the maximum
    // hardware-supported SIMD vector width.
    let Some(min_vector_ilp) = T::MIN_VECTORIZABLE_ILP else {
        return accumulators;
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
        *acc = pessimize::hide::<T>(*acc);
    }
    accumulators
}
