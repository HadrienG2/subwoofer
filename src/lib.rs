#![feature(portable_simd)]

pub mod arch;
pub mod process;
pub mod types;

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
