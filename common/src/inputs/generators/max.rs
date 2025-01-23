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

#[cfg(test)]
mod tests {
    use crate::{inputs::generators::tests::num_subnormals, tests::assert_panics};
    use proptest::prelude::*;
    use std::panic::AssertUnwindSafe;

    /// Test [`generate_max_inputs()`]
    fn target_and_num_subnormals() -> impl Strategy<Value = (Vec<f32>, usize)> {
        any::<Vec<f32>>().prop_flat_map(|target| {
            let target_len = target.len();
            (Just(target), num_subnormals(target_len))
        })
    }
    //
    proptest! {
        #[test]
        fn generate_max_inputs((mut target, num_subnormals) in target_and_num_subnormals()) {
            // Handle invalid subnormals count and generate benchmark inputs
            let generate = |target: &mut [_]| super::generate_max_inputs(target, &mut rand::thread_rng(), num_subnormals);
            if num_subnormals > target.len() {
                return assert_panics(AssertUnwindSafe(|| {
                    generate(&mut target);
                }));
            }
            generate(&mut target);

            // Check that the generated inputs meet our expectations
            let mut actual_subnormals = 0;
            for elem in target {
                if elem.is_subnormal() {
                    actual_subnormals += 1;
                } else {
                    prop_assert!(elem.is_normal());
                }
            }
            prop_assert_eq!(actual_subnormals, num_subnormals);
        }
    }
}
