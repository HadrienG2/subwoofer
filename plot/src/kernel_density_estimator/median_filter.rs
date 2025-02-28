//! Median filter implementation

use log::trace;
use numerical_multiset::NumericalMultiset;
use std::{
    cmp::Ordering,
    collections::VecDeque,
    fmt::{Debug, Display},
    num::NonZeroUsize,
};

/// A median filter for kernel density bandwidth autotuning
///
/// This median filter is a bit specialized for our needs, in particular it only
/// accepts windows with an odd number of points and it uses an algorithm that
/// only makes sense for floating-point elements (for integers, histogramming is
/// more efficient and should usually be preferred).
pub struct MedianFilter<T> {
    /// Sliding window over which the median is being computed
    ///
    /// Used to track the entry and exit of data points, which will then guide
    /// the update of the other fields.
    ///
    /// Window size must be an odd number >= 3.
    sliding_window: VecDeque<T>,

    /// Multiset of the window elements smaller than the median
    ///
    /// Used to count how many window elements are smaller than the median, and
    /// to replace the median with the largest of these elements when more than
    /// half of the window elements are located below the median.
    below_median: NumericalMultiset<T>,

    /// Current median window element
    median: T,

    /// Number of window elements equal to the current median
    num_median: usize,

    /// Multiset of the window elements larger than the median
    ///
    /// Plays the same role as `below_median`, but for window elements that are
    /// greater than the current median.
    above_median: NumericalMultiset<T>,
}
//
impl<T: Copy + Debug + Display + Ord> MedianFilter<T> {
    /// Initialize the median filter
    ///
    /// The initial window must contain an odd number of data points greater
    /// than or equal to 3. From this point on, the median filter will operate
    /// over this number of data points.
    ///
    /// The first point in the initial window is considered to be the oldest
    /// data point (and will thus be "forgotten" first), while the last point is
    /// considered to be the newest one (and will thus be "forgotten" last).
    ///
    /// # Panics
    ///
    /// If the initial window contains an even number of data points (which
    /// makes the computation much more expensive for a small result quality
    /// change, and is thus not supported) or a single data point (in which case
    /// it is pointless to go through a complex median computation).
    pub fn new(mut initial_window: Box<[T]>) -> Self {
        // Check precondition
        assert!(
            initial_window.len() % 2 == 1,
            "Medians with an even number of data points are unsupported"
        );
        assert!(
            initial_window.len() > 1,
            "Medians over a single data point are pointless"
        );

        // Set up the sliding windows
        let sliding_window = VecDeque::from(initial_window.to_vec());

        // Sort the initial window
        initial_window.sort_unstable();
        let sorted_window = initial_window;

        // Determine median value
        let half_window_len = sorted_window.len() / 2;
        let median = sorted_window[half_window_len];

        // Classify window elements as below median, median and above median
        let mut elements = sorted_window.into_vec().into_iter().peekable();
        //
        let mut below_median = NumericalMultiset::new();
        while let Some(elem) = elements.next_if(|&elem| elem < median) {
            below_median.insert(elem);
        }
        //
        let mut num_median = 0;
        while elements.next_if(|&elem| elem == median).is_some() {
            num_median += 1;
        }
        //
        let mut above_median = NumericalMultiset::new();
        while let Some(elem) = elements.next_if(|&elem| elem > median) {
            above_median.insert(elem);
        }
        assert!(elements.next().is_none());

        // ...and we're ready for the next data points
        Self {
            sliding_window,
            below_median,
            median,
            num_median,
            above_median,
        }
    }

    /// Current median value
    pub fn median(&self) -> T {
        self.median
    }

    /// Inject a new data point into the filter, removing the oldest point from
    /// the sliding window, and get the resulting new median
    #[inline]
    pub fn update(&mut self, input: T) -> T {
        // Check initial state in debug builds
        self.debug_check();

        // Update sliding window, note which point was removed
        let added = input;
        let removed = self
            .sliding_window
            .pop_front()
            .expect("Checked at construction that window contains >= 3 elements");
        self.sliding_window.push_back(added);

        // Describe update transaction
        trace!(
            "Starting from rolling median state...\n\
            - Below: {:#?}\n\
            - Median: {} ({} occurence(s))\n\
            - Above: {:#?}\n\
            ...will now proceed to remove data point {removed} and add data point {added}",
            self.below_median,
            self.median,
            self.num_median,
            self.above_median
        );

        // Update most median tracking state, check if this changes the median
        let half_window_len = self.sliding_window.len() / 2;
        let new_median_and_count = match (removed.cmp(&self.median), added.cmp(&self.median)) {
            (Ordering::Less, Ordering::Less) => {
                trace!("This only affects elements below the median => Median won't change");
                if added != removed {
                    self.below_median.remove(removed).expect(
                        "If we're removing a sample below the median, a sample should be present",
                    );
                    self.below_median.insert(added);
                }
                None
            }
            (Ordering::Less, Ordering::Equal) => {
                trace!("This increases median weight from below => Median won't change");
                self.below_median.remove(removed).expect(
                    "If we're removing a sample below the median, a sample should be present",
                );
                self.num_median += 1;
                None
            }
            (Ordering::Less, Ordering::Greater) => {
                trace!("This transfers weight from below the median to above");
                self.below_median.remove(removed).expect(
                    "If we're removing a sample below the median, a sample should be present",
                );
                self.above_median.insert(added);
                if self.above_median.len() > half_window_len {
                    debug_assert_eq!(self.above_median.len(), half_window_len + 1);
                    trace!(
                        "Weight above median {} has become greater than median weight {} + lower weight {} => Shift median up",
                        self.above_median.len(), self.num_median, self.below_median.len()
                    );
                    self.below_median.insert_multiple(
                        self.median,
                        NonZeroUsize::new(self.num_median)
                            .expect("Median should have >1 element associated with it"),
                    );
                    self.above_median.pop_all_first()
                } else {
                    None
                }
            }
            (Ordering::Equal, Ordering::Less) => {
                trace!("This transfers weight from the median to below");
                self.num_median -= 1;
                self.below_median.insert(added);
                if self.below_median.len() > half_window_len {
                    trace!(
                        "Weight below median {} has become greater than median weight {} + upper weight {} => Shift median down",
                        self.below_median.len(), self.num_median, self.above_median.len()
                    );
                    if let Some(num_median) = NonZeroUsize::new(self.num_median) {
                        self.above_median.insert_multiple(self.median, num_median);
                    }
                    self.below_median.pop_all_last()
                } else {
                    None
                }
            }
            (Ordering::Equal, Ordering::Equal) => {
                trace!("This replaces a median element with another => No change");
                None
            }
            (Ordering::Equal, Ordering::Greater) => {
                trace!("This transfers weight from the median to above");
                self.num_median -= 1;
                self.above_median.insert(added);
                if self.above_median.len() > half_window_len {
                    trace!(
                        "Weight above median {} has become greater than median weight {} + lower weight {} => Shift median up",
                        self.above_median.len(), self.num_median, self.below_median.len()
                    );
                    if let Some(num_median) = NonZeroUsize::new(self.num_median) {
                        self.below_median.insert_multiple(self.median, num_median);
                    }
                    self.above_median.pop_all_first()
                } else {
                    None
                }
            }
            (Ordering::Greater, Ordering::Less) => {
                trace!("This transfers weight from above the median to below");
                self.above_median.remove(removed).expect(
                    "If we're removing a sample above the median, a sample should be present",
                );
                self.below_median.insert(added);
                if self.below_median.len() > half_window_len {
                    // If enough weight shifts below the current median, the
                    // median will shift by one place below
                    trace!(
                        "Weight below median {} has become greater than median weight {} + upper weight {} => Shift median down",
                        self.below_median.len(), self.num_median, self.above_median.len()
                    );
                    self.above_median.insert_multiple(
                        self.median,
                        NonZeroUsize::new(self.num_median)
                            .expect("Median should have >1 element associated with it"),
                    );
                    self.below_median.pop_all_last()
                } else {
                    None
                }
            }
            (Ordering::Greater, Ordering::Equal) => {
                // Add weight to the median from above
                trace!("This increase median weight from above => Median won't change");
                self.above_median.remove(removed).expect(
                    "If we're removing a sample above the median, a sample should be present",
                );
                self.num_median += 1;
                None
            }
            (Ordering::Greater, Ordering::Greater) => {
                // Shuffle supra-median elements
                trace!("This only affects elements above the median => Median won't change");
                if added != removed {
                    self.above_median.remove(removed).expect(
                        "If we're removing a sample above the median, a sample should be present",
                    );
                    self.above_median.insert(added);
                }
                None
            }
        };

        // Store new median if any, then return final median
        if let Some((new_median, new_count)) = new_median_and_count {
            self.median = new_median;
            self.num_median = new_count.get();
        };
        self.median
    }

    // Check state integrity in debug builds
    fn debug_check(&self) {
        // Only do this in debug builds
        if !cfg!(debug_assertions) {
            return;
        }

        // Check conservation of number of data points by exploiting the fact
        // that VecDeque capacity does not grow in increments of 1
        debug_assert_eq!(self.sliding_window.len(), self.sliding_window.capacity());
        debug_assert_eq!(
            self.below_median.len() + self.num_median + self.above_median.len(),
            self.sliding_window.len()
        );

        // Check that the median does not have more than half of the elements
        // below or above it (otherwise it wouldn't be the median
        let half_window_len = self.sliding_window.len() / 2;
        debug_assert!(self.below_median.len() <= half_window_len);
        debug_assert!(self.above_median.len() <= half_window_len);

        // Check that the median has some data points associated with it
        debug_assert!(self.num_median > 0);

        // Check that the median meets expected ordering constraints
        if let Some((max_below, _multiplicity)) = self.below_median.last() {
            debug_assert!(self.median > max_below);
        }
        if let Some((min_above, _multiplicity)) = self.above_median.first() {
            debug_assert!(self.median < min_above);
        }
    }
}
