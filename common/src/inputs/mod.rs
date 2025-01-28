//! Benchmark input datasets

pub mod generators;

use crate::floats::FloatLike;

/// Owned or borrowed ordered set of benchmark inputs
///
/// We run benchmarks at all input sizes that are meaningful in hardware, from
/// inputs that can stay resident in a few CPU registers to arbitrarily large
/// in-memory datasets. But our benchmarking procedures need a few
/// size-dependent adaptations.
///
/// This trait abstracts over the two kind of inputs, and allows us to
/// manipulate them using an interface that is as homogeneous as possible.
pub trait Inputs: AsRef<[Self::Element]> {
    /// Floating-point input type
    type Element: FloatLike;

    /// Kind of input dataset, used for type-dependent computations
    const KIND: InputKind;

    /// Make the compiler think that this is an entirely different input, with a
    /// minimal impact on CPU register locality
    ///
    /// Implementations must be marked `#[inline]` to work as expected.
    fn hide_inplace(&mut self);

    /// Copy this data if owned, otherwise reborrow it as immutable
    ///
    /// This operation is needed in order to minimize the harmful side-effects
    /// of the `hide_inplace()` optimization barrier.
    ///
    /// It should be marked `#[inline]` so the compiler gets a better picture of
    /// the underlying benchmark data flow.
    fn freeze(&mut self) -> Self::Frozen<'_>;

    /// Clone of Self if possible, otherwise in-place reborrow
    type Frozen<'parent>: Inputs<Element = Self::Element>
    where
        Self: 'parent;
}
//
/// Like [`Inputs`] but allows unrestricted element mutation
pub trait InputsMut: Inputs + AsMut<[Self::Element]> {}

/// Kind of [`InputStorage`] that we are dealing with
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputKind {
    /// In-register dataset
    ///
    /// This input dataset is designed to be permanently held into the specified
    /// amount of CPU registers throughout the entire duration of a benchmark.
    ///
    /// This extra register pressure reduces the number of CPU registers that
    /// can be used as accumulators, and thus the degree of Instruction-Level
    /// Parallelism that can be used in the benchmark.
    ///
    /// To reduce this effect, inputs from an in-register dataset should be
    /// reused, i.e. each benchmark accumulator should be fed with all inputs
    /// from the dataset. This comes at the expense of reduced realism, and an
    /// increased potential for compiler over-optimization that may require
    /// stronger optimization barriers to mitigate.
    ReusedRegisters { count: usize },

    /// In-memory dataset
    ///
    /// This input dataset is designed to stay resident in one of the layers of
    /// the CPU's memory hierarchy, maybe a data cache, maybe only RAM.
    ///
    /// The associated storage is not owned by a particular benchmark, but
    /// borrowed from a longer-lived allocation.
    Memory,
}
//
impl InputKind {
    /// Truth that this kind of inputs is reused, i.e. each input from the
    /// dataset gets fed into each benchmark accumulator
    ///
    /// Otherwise, each benchmark accumulator gets fed with its own subset of
    /// the input data set.
    pub const fn is_reused(self) -> bool {
        match self {
            Self::ReusedRegisters { .. } => true,
            Self::Memory => false,
        }
    }
}

// Implementations of Inputs/InputsMut
impl<T: FloatLike, const N: usize> Inputs for [T; N] {
    type Element = T;

    const KIND: InputKind = InputKind::ReusedRegisters { count: N };

    #[inline]
    fn hide_inplace(&mut self) {
        for elem in self {
            let old_elem = *elem;
            let new_elem = pessimize::hide::<T>(old_elem);
            *elem = new_elem;
        }
    }

    #[inline]
    fn freeze(&mut self) -> Self::Frozen<'_> {
        *self
    }

    type Frozen<'parent>
        = [T; N]
    where
        T: 'parent;
}
//
impl<T: FloatLike, const N: usize> InputsMut for [T; N] {}
//
impl<'buffer, T: FloatLike> Inputs for &'buffer [T] {
    type Element = T;

    const KIND: InputKind = InputKind::Memory;

    #[inline]
    fn hide_inplace<'hidden>(&'hidden mut self) {
        *self =
            // SAFETY: Although the borrow checker does not know it,
            //         pessimize::hide(*self) is just *self, so this is a no-op
            //         *self = *self instruction that is safe by definition.
            unsafe { std::mem::transmute::<&'hidden [T], &'buffer [T]>(pessimize::hide(*self)) }
    }

    #[inline]
    fn freeze(&mut self) -> Self::Frozen<'_> {
        self
    }

    type Frozen<'parent>
        = &'parent [T]
    where
        Self: 'parent;
}
//
impl<'buffer, T: FloatLike> Inputs for &'buffer mut [T] {
    type Element = T;

    const KIND: InputKind = InputKind::Memory;

    #[inline]
    fn hide_inplace<'hidden>(&'hidden mut self) {
        *self =
            // SAFETY: Although the borrow checker does not know it,
            //         pessimize::hide(*self) is just *self, so this is a no-op
            //         *self = *self instruction that is safe by definition.
            unsafe { std::mem::transmute::<&'hidden mut [T], &'buffer mut [T]>(pessimize::hide(*self)) }
    }

    #[inline]
    fn freeze(&mut self) -> Self::Frozen<'_> {
        self
    }

    type Frozen<'parent>
        = &'parent [T]
    where
        Self: 'parent;
}
//
impl<T: FloatLike> InputsMut for &mut [T] {}

/// Test utilities that are used by other crates in the workspace
#[cfg(feature = "unstable_test")]
pub mod test_utils {
    use proptest::{prelude::*, sample::SizeRange};

    /// Generate an arbitrary f32 (including NaN)
    fn any_f32() -> impl Strategy<Value = f32> {
        use prop::num::f32::*;
        POSITIVE | NEGATIVE | NORMAL | SUBNORMAL | ZERO | INFINITE | QUIET_NAN
    }

    /// Generate a array of f32s
    pub fn f32_array<const SIZE: usize>() -> impl Strategy<Value = [f32; SIZE]> {
        std::array::from_fn(|_| any_f32())
    }

    /// Generate a Vec of f32s
    pub fn f32_vec() -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(any_f32(), SizeRange::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{prelude::*, sample::SizeRange};

    proptest! {
        #[test]
        fn input_kind(kind: InputKind) {
            match kind {
                InputKind::ReusedRegisters { .. } => prop_assert!(kind.is_reused()),
                InputKind::Memory => prop_assert!(!kind.is_reused()),
            }
        }
    }

    /// Test implementations of the [`Inputs`] trait
    fn test_inputs<I: Inputs>(
        mut inputs: I,
        expected: impl AsRef<[I::Element]>,
        expected_reused: bool,
    ) -> Result<(), TestCaseError> {
        let expected = expected.as_ref();
        if expected_reused {
            prop_assert_eq!(
                I::KIND,
                InputKind::ReusedRegisters {
                    count: expected.len()
                }
            );
        } else {
            prop_assert_eq!(I::KIND, InputKind::Memory);
        }
        prop_assert_eq!(inputs.as_ref(), expected);
        inputs.hide_inplace();
        prop_assert_eq!(inputs.as_ref(), expected);
        let frozen = inputs.freeze();
        prop_assert_eq!(frozen.as_ref(), expected);
        Ok(())
    }
    //
    macro_rules! test_inputs_for {
        ($($t:ident),*) => ($(
            mod $t {
                use super::*;

                #[test]
                fn inputs_0regs() {
                    test_inputs::<[$t; 0]>([], [], true).unwrap();
                }

                fn non_nan() -> impl Strategy<Value = $t> {
                    use prop::num::$t::*;
                    POSITIVE | NEGATIVE | NORMAL | SUBNORMAL | ZERO | INFINITE
                }
                //
                fn non_nan_array<const SIZE: usize>() -> impl Strategy<Value = [$t; SIZE]> {
                    std::array::from_fn(|_| non_nan())
                }
                //
                proptest! {
                    #[test]
                    fn inputs_1reg(regs in non_nan_array::<1>()) {
                        test_inputs(regs, regs, true)?;
                    }

                    #[test]
                    fn inputs_2regs(regs in non_nan_array::<2>()) {
                        test_inputs(regs, regs, true)?;
                    }

                    #[test]
                    fn inputs_3regs(regs in non_nan_array::<3>()) {
                        test_inputs(regs, regs, true)?;
                    }

                    #[test]
                    fn inputs_4regs(regs in non_nan_array::<4>()) {
                        test_inputs(regs, regs, true)?;
                    }

                    #[test]
                    fn inputs_mem(mem in prop::collection::vec(non_nan(), SizeRange::default())) {
                        test_inputs(&mem[..], &mem[..], false)?;
                    }
                }
            }
        )*);
    }
    //
    test_inputs_for!(f32, f64);

    /// Assert that a type implements the [`InputsMut`] trait
    fn assert_inputs_mut<I: InputsMut>() {}
    //
    #[test]
    fn inputs_mut() {
        assert_inputs_mut::<[f32; 0]>();
        assert_inputs_mut::<[f32; 1]>();
        assert_inputs_mut::<[f32; 2]>();
        assert_inputs_mut::<[f32; 3]>();
        assert_inputs_mut::<[f32; 4]>();
        assert_inputs_mut::<[f64; 0]>();
        assert_inputs_mut::<[f64; 1]>();
        assert_inputs_mut::<[f64; 2]>();
        assert_inputs_mut::<[f64; 3]>();
        assert_inputs_mut::<[f64; 4]>();
        assert_inputs_mut::<&mut [f32]>();
        assert_inputs_mut::<&mut [f64]>();
    }
}
