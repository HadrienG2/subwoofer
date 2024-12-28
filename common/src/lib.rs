//! Common infrastructure shared by all benchmark workloads

#![cfg_attr(feature = "simd", feature(portable_simd))]

pub mod arch;
pub mod floats;
pub mod inputs;
pub mod operations;
