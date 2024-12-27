#![allow(unused)]
#![feature(portable_simd)]

use criterion::{criterion_group, criterion_main, Criterion};
use std::simd::f32x4;
use subwoofer::TypeBenchmark;

pub fn criterion_benchmark(criterion: &mut Criterion) {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse"
    ))]
    TypeBenchmark::new(criterion).benchmark_type::<f32x4>("f32x04");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
