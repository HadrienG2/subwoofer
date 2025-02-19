//! Ordered multiset implementation

use std::{
    collections::btree_map::{BTreeMap, Entry},
    num::NonZeroUsize,
};

/// An ordered multiset for floating-point median filtering
///
/// This data structure maintains a multiset of values. It is a bit like a
/// mathematical set, but instead of enforcing value uniqueness we track how
/// many occurences of a contained value exists. Furthermore, this multiset
/// implementation is ordered, which allows one can cheaply query what are the
/// minimal and maximal values contained inside of the set.
///
/// We use this multiset for median filtering since a pair of those multisets
/// allows us to cheaply track values below and above the current median, count
/// how many values there are on each side, and easily replace the median by the
/// value immediately below or above it (max of values below and min of values
/// above respectively) when it stops being the median.
///
/// Note that performing median filtering like this only makes sense for
/// floating-point numbers and large integers. For small integers, one
/// should prefer histogram-based algorithms due to their superior efficiency.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OrderedMultiset<T> {
    set: BTreeMap<T, NonZeroUsize>,
    len: usize,
}
//
impl<T> OrderedMultiset<T> {
    /// Build an empty multiset
    #[must_use = "Only produces a result"]
    pub fn new() -> Self {
        Self {
            set: BTreeMap::new(),
            len: 0,
        }
    }

    /// Truth that this set contains no elements
    ///
    /// Algorithm complexity does not depend on multiset content.
    #[must_use = "Only produces a result"]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of values currently present in the set
    ///
    /// Algorithm complexity does not depend on multiset content.
    #[must_use = "Only produces a result"]
    pub fn len(&self) -> usize {
        self.len
    }
}
//
impl<T: Copy + Ord> OrderedMultiset<T> {
    /// Minimal value present in the set
    ///
    /// Algorithm complexity is `O(log(Nkeys))`.
    #[must_use = "Only produces a result"]
    pub fn min(&self) -> Option<T> {
        self.set.first_key_value().map(|(&k, _v)| k)
    }

    /// Maximal value present in the set
    ///
    /// Algorithm complexity is `O(log(Nkeys))`.
    #[must_use = "Only produces a result"]
    pub fn max(&self) -> Option<T> {
        self.set.last_key_value().map(|(&k, _v)| k)
    }
}
//
impl<T: Ord> OrderedMultiset<T> {
    /// Insert a value, tell how many identical values were already present in
    /// the multiset before insertion
    ///
    /// Algorithm complexity is `O(log(Nkeys))`.
    #[inline]
    pub fn insert(&mut self, value: T) -> Option<NonZeroUsize> {
        self.insert_multiple(value, NonZeroUsize::new(1).unwrap())
    }

    /// Insert multiple copies of a value, tell how many identical values were
    /// already present in the multiset before insertion
    ///
    /// Algorithm complexity is `O(log(Nkeys))`.
    #[inline]
    pub fn insert_multiple(&mut self, value: T, count: NonZeroUsize) -> Option<NonZeroUsize> {
        let result = match self.set.entry(value) {
            Entry::Vacant(v) => {
                v.insert(count);
                None
            }
            Entry::Occupied(mut o) => {
                let old_count = *o.get();
                *o.get_mut() = old_count
                    .checked_add(count.get())
                    .expect("Multiset duplicate count has overflown");
                Some(old_count)
            }
        };
        self.len += count.get();
        result
    }

    /// Attempt to remove one occurence of a value from the multiset, on success
    /// tell how many values were previously present in the multiset
    ///
    /// Algorithm complexity is `O(log(Nkeys))`.
    #[inline]
    #[must_use = "Invalid removal should be handled"]
    pub fn remove(&mut self, value: T) -> Option<NonZeroUsize> {
        let result = match self.set.entry(value) {
            Entry::Vacant(_) => None,
            Entry::Occupied(mut o) => {
                let old_value = *o.get();
                match NonZeroUsize::new(old_value.get() - 1) {
                    Some(new_value) => {
                        *o.get_mut() = new_value;
                    }
                    None => {
                        o.remove_entry();
                    }
                }
                Some(old_value)
            }
        };
        self.len -= 1;
        result
    }

    /// Remove all copies of the smallest value from the set, if any
    ///
    /// Algorithm complexity is `O(log(Nkeys))`.
    #[must_use = "Invalid removal should be handled"]
    pub fn remove_all_min(&mut self) -> Option<(T, NonZeroUsize)> {
        self.set
            .pop_first()
            .inspect(|(_value, count)| self.len -= count.get())
    }

    /// Remove all copies of the largest value from the set, if any
    ///
    /// Algorithm complexity is `O(log(Nkeys))`.
    #[must_use = "Invalid removal should be handled"]
    pub fn remove_all_max(&mut self) -> Option<(T, NonZeroUsize)> {
        self.set
            .pop_last()
            .inspect(|(_value, count)| self.len -= count.get())
    }
}
//
impl<T> OrderedMultiset<T> {
    /// Remove all contents from this multiset
    ///
    /// Algorithm complexity does not depend on multiset content.
    pub fn clear(&mut self) {
        self.set.clear();
        self.len = 0;
    }
}
//
impl<T: Ord> FromIterator<T> for OrderedMultiset<T> {
    #[must_use = "Only produces a result"]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut result = Self::new();
        for value in iter {
            result.insert(value);
        }
        result
    }
}
