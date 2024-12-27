use common::floats::FloatLike;
use criterion::Criterion;

pub fn benchmark_type<T: FloatLike>(criterion: &mut Criterion) {}
