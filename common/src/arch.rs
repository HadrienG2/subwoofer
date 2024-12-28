//! Miscellaneous CPU microarchitecture information

use target_features::Architecture;

/// Lower bound on the number of architectural scalar/SIMD registers.
///
/// To exhaustively cover all hardware-allowed configurations, this should
/// ideally strive to be an exact count, not a lower bound, for all hardware of
/// highest interest to this benchmark's users.
///
/// However, in the case of hardware architectures like Arm that have a
/// complicated past and a fragmented present, accurately spelling this out for
/// all hardware that can be purchased today would be quite a bit of effort with
/// relatively little payoff. So we tolerate some lower bound approximation.
pub const MIN_FLOAT_REGISTERS: usize = const {
    let target = target_features::CURRENT_TARGET;
    match target.architecture() {
        Architecture::Arm | Architecture::AArch64 => {
            if target.supports_feature_str("sve") {
                32
            } else {
                16
            }
        }
        Architecture::X86 => {
            if target.supports_feature_str("avx512vl") {
                32
            } else {
                16
            }
        }
        Architecture::RiscV => 32,
        // TODO: Check for other architectures
        _ => 16,
    }
};

/// Truth that the current hardware architecture is known to support memory
/// operands for scalar and SIMD operations.
///
/// This means that when we are doing benchmarks like `addsub` that directly
/// reduce memory inputs into accumulators, we don't need to load inputs into
/// CPU registers before reducing them into the accumulator. As a result, we can
/// use more CPU registers as accumulators on those benchmarks.
pub const HAS_MEMORY_OPERANDS: bool = const {
    let target = target_features::CURRENT_TARGET;
    match target.architecture() {
        Architecture::X86 => true,
        // TODO: Check for other architectures
        _ => false,
    }
};

/// Truth that the current hardware architecture is known to have native FMA
pub const HAS_HARDWARE_FMA: bool = const {
    let target = target_features::CURRENT_TARGET;
    match target.architecture() {
        Architecture::X86 => target.supports_feature_str("fma"),
        // TODO: Check for other architectures
        _ => false,
    }
};
