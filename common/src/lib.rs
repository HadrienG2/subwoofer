//! Common infrastructure shared by all benchmark workloads

#![cfg_attr(feature = "simd", feature(portable_simd))]

pub mod arch;
pub mod floats;
pub mod inputs;
pub mod operations;

#[cfg(test)]
pub(crate) mod tests {
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
