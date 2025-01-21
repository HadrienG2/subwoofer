//! Input generator for `acc <- max(acc, f(input))` benchmarks

use crate::floats::{self, FloatLike};
use rand::prelude::*;

/// Generate a mixture of normal and subnormal inputs for a benchmark that
/// follows the `acc -> max(acc, f(input))` pattern, where the accumulator is
/// initially a normal number and f is a function that turns any positive normal
/// number into another positive normal number.
///
/// These benchmarks can tolerate any sequence of normal and subnormal numbers,
/// without any risk of overflow or cancelation. Hence they get the simplest and
/// most general input generation procedure.
pub fn generate_max_inputs<T: FloatLike, R: Rng>(
    target: &mut [T],
    rng: &mut R,
    num_subnormals: usize,
) {
    // Split the target slice into one part for normal numbers and another part
    // for subnormal numbers
    assert!(num_subnormals <= target.len());
    let (subnormals, normals) = target.split_at_mut(num_subnormals);

    // Generate the subnormal inputs
    let subnormal = floats::subnormal_sampler();
    for elem in subnormals {
        *elem = subnormal(rng);
    }

    // Generate the normal inputs
    let normal = floats::normal_sampler();
    for elem in normals {
        *elem = normal(rng);
    }

    // Randomize the order of normal and subnormal inputs
    target.shuffle(rng)
}
