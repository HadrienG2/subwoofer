//! Common infrastructure shared by all benchmark workloads

#![cfg_attr(feature = "simd", feature(portable_simd))]

pub mod arch;
pub mod floats;
pub mod inputs;
pub mod operations;

#[cfg(feature = "unstable_test")]
pub use proptest;

#[cfg(any(test, feature = "unstable_test"))]
pub mod test_utils {
    use proptest::prelude::*;
    use std::panic::{self, UnwindSafe};

    /// Assert that a function panics in a proptest-friendly manner
    pub fn assert_panics<T>(f: impl FnOnce() -> T + UnwindSafe) -> Result<(), TestCaseError> {
        if panic::catch_unwind(f).is_err() {
            Ok(())
        } else {
            Err(TestCaseError::fail("this function should have panicked"))
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    /// Number of tests to execute for "hidden" property-based tests (i.e. those
    /// that are property based but do not use proptest's RNG)
    ///
    /// Attempts to match the match the behavior of proptest for consistency.
    pub fn proptest_cases() -> usize {
        std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|cases| cases.parse().ok())
            .unwrap_or(256)
    }
}
