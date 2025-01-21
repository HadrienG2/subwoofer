//! Input generation for benchmarks with an ADD/SUB pattern

use super::{DataStream, GeneratorStream, InputGenerator};
use crate::{floats::FloatLike, inputs::InputsMut};
use rand::prelude::*;

/// Generate a mixture of normal and subnormal inputs for a benchmark that
/// follows the `acc -> acc + input * constant` pattern where both `constant`
/// and the initial accumulator value are positive numbers in the usual "narrow"
/// range `[0.5; 2[`.
///
/// Overall, the general strategy to avoid unbounded accumulator growth or
/// cancelation is that...
///
/// - Generated normal inputs are in the same magnitude range as the initial
///   accumulator value `[0.5; 2[`
/// - Every time we add a normal value to the accumulator, we subtract it back
///   on the next normal input, thusly bringing the accumulator back to its
///   initial value (except for a few zeroed low-order mantissa bits)
/// - In case the number of normal inputs is odd, we cannot do this with the
///   last input, so we set it to `0` instead
pub fn generate_add_inputs<Storage: InputsMut, const ILP: usize>(
    target: &mut Storage,
    rng: &mut impl Rng,
    num_subnormals: usize,
) {
    super::generate_input_streams::<_, _, ILP>(target, rng, num_subnormals, AddGenerator);
}

/// (Lack of) global state for this input generator
struct AddGenerator;
//
impl<T: FloatLike, R: Rng> InputGenerator<T, R> for AddGenerator {
    type Stream<'a> = AddStream<T>;

    fn new_stream(&self) -> Self::Stream<'_> {
        AddStream::Unconstrained
    }
}

/// Per-stream state for this input generator
enum AddStream<T: FloatLike> {
    /// No constraint on the next normal value
    Unconstrained,

    /// Next normal value should negate the effect of the previous one
    Negate {
        /// Previous normal input value
        value: T,

        /// Global position of this normal value in the generated data
        global_idx: usize,
    },
}
//
impl<T: FloatLike, R: Rng> GeneratorStream<R> for AddStream<T> {
    type Float = T;

    #[inline]
    fn record_subnormal(&mut self) {}

    #[inline]
    fn generate_normal(
        &mut self,
        global_idx: usize,
        rng: &mut R,
        mut narrow: impl FnMut(&mut R) -> T,
    ) -> T {
        match *self {
            Self::Unconstrained => {
                let value = narrow(rng);
                *self = Self::Negate { global_idx, value };
                value
            }
            Self::Negate { value, .. } => {
                *self = Self::Unconstrained;
                -value
            }
        }
    }

    fn finalize(self, mut stream: DataStream<'_, T>) {
        if let Self::Negate { global_idx, .. } = self {
            *stream.scalar_at(global_idx) = T::splat(0.0);
        }
    }
}
