//! Miscellaneous CPU microarchitecture information

use target_features::Architecture;

/// Truth that the current CPU ISA allows emitting the result of a division in
/// the register that used to hold the denominator
pub const ALLOWS_DIV_OUTPUT_DENOMINATOR: bool = const {
    let target = target_features::CURRENT_TARGET;
    match target.architecture() {
        Architecture::X86 => target.supports_feature_str("avx"),
        // TODO: Check for other architectures
        _ => true,
    }
};

/// Truth that the current CPU ISA allows using a memory operand for the
/// numerator of a division. Implies [`HAS_MEMORY_OPERANDS`].
pub const ALLOWS_DIV_MEMORY_NUMERATOR: bool = const {
    let target = target_features::CURRENT_TARGET;
    match target.architecture() {
        // As of AVX-512 at least, x86 will not allow this
        Architecture::X86 => false,
        // TODO: Check for other architectures
        _ => HAS_MEMORY_OPERANDS,
    }
};

/// Truth that the current CPU ISA has a fused multiply-add instruction
pub const HAS_HARDWARE_FMA: bool = const {
    let target = target_features::CURRENT_TARGET;
    match target.architecture() {
        Architecture::X86 => target.supports_feature_str("fma"),
        // TODO: Check for other architectures
        _ => false,
    }
};

/// Truth that the current CPU ISA has an FNMA instruction. Implies
/// [`HAS_HARDWARE_FMA`].
///
/// FNMA is to FMA what `-a * b + c` is to `a * b + c`.
pub const HAS_HARDWARE_NEGATED_FMA: bool = const {
    let target = target_features::CURRENT_TARGET;
    match target.architecture() {
        Architecture::X86 => target.supports_feature_str("fma"),
        // TODO: Check for other architectures
        _ => false,
    }
};

/// Truth that the CPU ISA is known to support memory operands for scalar and
/// SIMD operations.
///
/// This means that when we are doing benchmarks like `add` that directly reduce
/// memory inputs into accumulators, we don't need to load inputs into CPU
/// registers before reducing them into the accumulator. This reduces register
/// pressure, so we can use more CPU registers as accumulators on those some
/// benchmarks.
pub const HAS_MEMORY_OPERANDS: bool = const {
    let target = target_features::CURRENT_TARGET;
    match target.architecture() {
        Architecture::X86 => true,
        // TODO: Check for other architectures
        _ => false,
    }
};

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
                // Glossing over the VFP mess here, since this is a lower bound
                16
            }
        }
        Architecture::X86 => {
            // Technically we need avx512f for 512-bit registers and avx512vl
            // for other register widths, but as of today there are no CPUs that
            // support avx512f without supporting avx512vl, so there is no need
            // to make MIN_FLOAT_REGISTERS depend on the register type yet.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// When a property should imply another, make sure this is actually true
    #[allow(clippy::assertions_on_constants)]
    #[test]
    fn implications() {
        assert!(!ALLOWS_DIV_MEMORY_NUMERATOR || HAS_MEMORY_OPERANDS);
        assert!(!HAS_HARDWARE_NEGATED_FMA || HAS_HARDWARE_FMA);
    }
}
