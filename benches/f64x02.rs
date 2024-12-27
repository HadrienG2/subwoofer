#![allow(unused)]
#![feature(portable_simd)]

use criterion::{criterion_group, criterion_main, Criterion};
use std::simd::f64x2;
use subwoofer::TypeBenchmark;

pub fn criterion_benchmark(criterion: &mut Criterion) {
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse2"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    TypeBenchmark::new(criterion).benchmark_type::<f64x2>("f64x02");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
