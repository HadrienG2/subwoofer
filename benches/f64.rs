use criterion::{criterion_group, criterion_main, Criterion};
use subwoofer::TypeBenchmark;

pub fn criterion_benchmark(criterion: &mut Criterion) {
    TypeBenchmark::new(criterion).benchmark_type::<f64>("f64");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
