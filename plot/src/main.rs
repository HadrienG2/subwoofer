use anyhow::{bail, Context, Result};
use criterion::Throughput;
use criterion_cbor::{DataDirectory, Search};
use log::{debug, info, trace};
use ordered_float::NotNan;
use std::{
    cmp::Ordering,
    collections::{btree_map::Entry, BTreeMap, VecDeque},
    num::NonZeroUsize,
    path::Path,
};

/// Length of the sliding window used to determine the kernel density
/// estimator's kernel bandwidth via median filtering
///
/// This is a fraction of the total dataset width, so a higher value means a
/// smaller window that spans less points, leading to a kernel density estimator
/// that looks less smooth but catches narrow peaks in the probability density
/// better.
const REL_HALF_SMOOTHING_LEN: usize = 10;

fn main() -> Result<()> {
    // Set up logging
    env_logger::init();

    // Locate workspace root
    let mut workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    while !workspace_root.join("Cargo.lock").exists() {
        workspace_root = workspace_root
            .parent()
            .context("Failed to locate workspace root")?;
    }

    // Determine where output data should be stored
    let mut plots_root = workspace_root.to_owned();
    plots_root.push("target");
    plots_root.push("plots");
    std::fs::create_dir_all(&plots_root).context("Failed to create output directory")?;

    // FIXME: Starting with just a few benchmark configurations
    let benchmark_filter = |dir: DataDirectory| {
        let dir_name = dir.dir_name();
        match dir.depth() {
            1 => {
                let Some(op_ilp_source) = dir_name.strip_prefix("f32x08_") else {
                    debug!("Ignoring benchmarks from {dir_name}: Input is not f32/AVX");
                    return false;
                };
                let Some(op_ilp_cache) = op_ilp_source.strip_suffix("cache") else {
                    debug!("Ignoring benchmarks from {dir_name}: Input is not from a CPU cache");
                    return false;
                };
                let (op_ilp_l, cache_level) = op_ilp_cache.split_at(op_ilp_cache.len() - 1);
                assert!(("1"..="9").contains(&cache_level));
                let Some(op_ilp) = op_ilp_l.strip_suffix("_L") else {
                    unreachable!(
                        "Directory name {dir_name} doesn't end with a cache level as expected"
                    );
                };
                let Some(op) = op_ilp.strip_suffix("_chained") else {
                    debug!("Ignoring benchmarks from {dir_name}: Benchmark isn't superscalar");
                    return false;
                };
                if op != "mul_max" {
                    debug!(
                        "Ignoring benchmarks from {dir_name}: Benchmark isn't about multiplication"
                    );
                    return false;
                }
                true
            }
            2 => {
                assert!(dir_name.ends_with('%'));
                if dir_name != "050.0%" {
                    debug!(
                        "Ignoring benchmark from {}: Input is not half-subnormal",
                        dir.path_from_data_root().display()
                    );
                    return false;
                }
                true
            }
            _ => unreachable!(
                "Unexpected benchmark path {}",
                dir.path_from_data_root().display()
            ),
        }
    };

    // Process benchmarks
    for bench in Search::in_cargo_root(workspace_root).find_in_paths(benchmark_filter) {
        // Check benchmark properties
        let bench = bench.context("Failed to enumerate benchmarks")?;
        info!(
            "Processing benchmark from {}",
            bench.path_from_data_root().display()
        );
        let metadata = bench.metadata().context("Failed to read benchmark data")?;
        info!("Benchmark has metadata {metadata:?}");

        // Load latest measurement
        let latest = metadata.latest_local_datetime();
        let Some(measurement) = bench
            .measurements()
            .find(|meas| meas.local_datetime() == latest)
        else {
            bail!("Benchmark metadata reports a latest measurement from {latest:?} but none found");
        };
        info!("Processing latest measurement from time {latest:?}");
        let data = measurement
            .data()
            .context("Failed to load latest measurement")?;

        // Compute the iteration times per element
        let Some(Throughput::Elements(elems_per_iter)) = data.throughput else {
            bail!("Unexpected throughput unit {:?}", data.throughput);
        };
        let throughput_norm = 1.0 / elems_per_iter as f64;
        let mut time_per_elem = data
            .avg_values
            .iter()
            .map(|avg| {
                NotNan::new(avg * throughput_norm).context("No NaN expected in timing measurements")
            })
            .collect::<Result<Box<[_]>>>()?;

        // Sort them to make statistical computations easier
        time_per_elem.sort_unstable();

        // Determine per-point kernel bandwidth for kernel density estimator
        let kernel_bandwidth = kernel_bandwidth(&time_per_elem);
        assert_eq!(kernel_bandwidth.len(), time_per_elem.len());

        // TODO: Compute kernel density estimator, using a band-limited and
        // cheap kernel function (Epanechnikov kernel) + make sure we only sum
        // points within the nonzero range of the previous points.

        todo!()
    }
    Ok(())
}

/// Determine the bandwidth of each data point in the kernel density estimator
fn kernel_bandwidth(sorted_samples: &[NotNan<f64>]) -> Box<[NotNan<f64>]> {
    debug_assert!(sorted_samples.is_sorted());

    // Compute the distance between consecutive sorted samples
    let raw_distances = sorted_samples
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect::<Box<[_]>>();
    debug!("Computing smoothed probability density kernel bandwidth from raw inter-sample distances {raw_distances:?}");

    // Prepare output storage
    let mut kernel_bandwidth = Vec::with_capacity(sorted_samples.len());

    // Set up sliding median filter
    let half_smoothing_len = raw_distances.len().div_ceil(REL_HALF_SMOOTHING_LEN);
    let full_smoothing_len = 2 * half_smoothing_len + 1;
    //
    let initial_window = raw_distances[..full_smoothing_len].to_owned();
    let mut initial_sorted = initial_window.clone();
    initial_sorted.sort_unstable();
    let mut window = VecDeque::from(initial_window);
    //
    let mut median = initial_sorted[half_smoothing_len];
    let initial_sorted = || initial_sorted.iter().copied();
    let mut below_median = initial_sorted()
        .take_while(|&x| x < median)
        .collect::<OrderedMultiset<_>>();
    let mut median_count = initial_sorted()
        .skip_while(|&x| x < median)
        .take_while(|&x| x == median)
        .count();
    let mut above_median = initial_sorted()
        .skip_while(|&x| x <= median)
        .collect::<OrderedMultiset<_>>();

    // Take first N kernel widths equal to first computable kernel width
    trace!(
        "Taking first {} distances equal to first window median {}",
        half_smoothing_len + 1,
        median
    );
    for _ in 0..=half_smoothing_len {
        kernel_bandwidth.push(median);
    }

    // Compute next kernel widths using sliding median filter
    for &added in raw_distances.iter().skip(full_smoothing_len) {
        // Update sliding window
        let removed = window
            .pop_front()
            .expect("by definition, window contains at least three elements");
        window.push_back(added);

        // Sliding median integrity check:
        trace!(
            "Starting from rolling median state...\n\
            - Below: {below_median:#?}\n\
            - Median: {median} ({median_count} occurence(s))\n\
            - Above: {above_median:#?}\n\
            ...will now proceed to remove data point {removed} and add data point {added}"
        );
        // - The sliding window population cannot change over time
        debug_assert_eq!(
            below_median.len() + median_count + above_median.len(),
            full_smoothing_len
        );
        // - The median should not have more than half of the sliding window
        //   above or below it.
        debug_assert!(below_median.len() <= half_smoothing_len);
        debug_assert!(above_median.len() <= half_smoothing_len);
        // - The median should have some elements associated with it
        debug_assert!(median_count > 0);
        // - The median should respect expected ordering
        if cfg!(debug_assertions) {
            if let Some(max_below) = below_median.max() {
                debug_assert!(median > max_below);
            }
            if let Some(min_above) = above_median.max() {
                debug_assert!(median < min_above);
            }
        }

        // Update sliding median
        let new_median_and_count = match (removed.cmp(&median), added.cmp(&median)) {
            (Ordering::Less, Ordering::Less) => {
                // Shuffle sub-median elements
                trace!("This only affects elements below the median => Median won't change");
                if added != removed {
                    below_median.remove(removed).expect(
                        "If we're removing a sample below the median, a sample should be present",
                    );
                    below_median.insert(added);
                }
                None
            }
            (Ordering::Less, Ordering::Equal) => {
                // Add weight to the median from below
                trace!("This adds weight to the median from below => Median won't change");
                below_median.remove(removed).expect(
                    "If we're removing a sample below the median, a sample should be present",
                );
                median_count += 1;
                None
            }
            (Ordering::Less, Ordering::Greater) => {
                // Transfer weight from below the median to above
                trace!("This transfers weight from below the median to above => Median MAY change");
                below_median.remove(removed).expect(
                    "If we're removing a sample below the median, a sample should be present",
                );
                above_median.insert(added);
                if above_median.len() > half_smoothing_len {
                    // If enough weight shifts above the current median, the
                    // median will shift by one place above
                    debug_assert_eq!(above_median.len(), half_smoothing_len + 1);
                    trace!(
                        "Weight above median {} has become greater than median weight {} + lower weight {} => Shift to higher median",
                        above_median.len(), median_count, below_median.len()
                    );
                    below_median.insert_several(
                        median,
                        NonZeroUsize::new(median_count)
                            .expect("Median should have >1 element associated with it"),
                    );
                    above_median.remove_all_min()
                } else {
                    None
                }
            }
            (Ordering::Equal, Ordering::Less) => {
                // Transfer weight from the median to below
                trace!("This transfers weight from the median to below => Median MAY change");
                median_count -= 1;
                below_median.insert(added);
                if below_median.len() > half_smoothing_len {
                    // If enough weight shifts below the current median, the
                    // median will shift by one place below
                    trace!(
                        "Weight below median {} has become greater than median weight {} + upper weight {} => Shift to lower median",
                        below_median.len(), median_count, above_median.len()
                    );
                    if let Some(median_count) = NonZeroUsize::new(median_count) {
                        above_median.insert_several(median, median_count);
                    }
                    below_median.remove_all_max()
                } else {
                    None
                }
            }
            (Ordering::Equal, Ordering::Equal) => {
                // Remove a median element and add it back => No change
                trace!("This replaces a median element with another => Nothing will change");
                None
            }
            (Ordering::Equal, Ordering::Greater) => {
                // Transfer weight from the median to above
                trace!("This transfers weight from the median to above => Median MAY change");
                median_count -= 1;
                above_median.insert(added);
                if above_median.len() > half_smoothing_len {
                    // If enough weight shifts above the current median, the
                    // median will shift by one place above
                    trace!(
                        "Weight above median {} has become greater than median weight {} + lower weight {} => Shift to higher median",
                        above_median.len(), median_count, below_median.len()
                    );
                    if let Some(median_count) = NonZeroUsize::new(median_count) {
                        below_median.insert_several(median, median_count);
                    }
                    above_median.remove_all_min()
                } else {
                    None
                }
            }
            (Ordering::Greater, Ordering::Less) => {
                // Transfer weight from above the median to below
                trace!("This transfers weight from above the median to below => Median MAY change");
                above_median.remove(removed).expect(
                    "If we're removing a sample above the median, a sample should be present",
                );
                below_median.insert(added);
                if below_median.len() > half_smoothing_len {
                    // If enough weight shifts below the current median, the
                    // median will shift by one place below
                    trace!(
                        "Weight below median {} has become greater than median weight {} + upper weight {} => Shift to lower median",
                        below_median.len(), median_count, above_median.len()
                    );
                    above_median.insert_several(
                        median,
                        NonZeroUsize::new(median_count)
                            .expect("Median should have >1 element associated with it"),
                    );
                    below_median.remove_all_max()
                } else {
                    None
                }
            }
            (Ordering::Greater, Ordering::Equal) => {
                // Add weight to the median from above
                trace!("This adds weight to the median from above => Median won't change");
                above_median.remove(removed).expect(
                    "If we're removing a sample above the median, a sample should be present",
                );
                median_count += 1;
                None
            }
            (Ordering::Greater, Ordering::Greater) => {
                // Shuffle supra-median elements
                trace!("This only affects elements above the median => Median won't change");
                if added != removed {
                    above_median.remove(removed).expect(
                        "If we're removing a sample above the median, a sample should be present",
                    );
                    above_median.insert(added);
                }
                None
            }
        };
        if let Some((new_median, new_count)) = new_median_and_count {
            median = new_median;
            median_count = new_count.get();
        }
        kernel_bandwidth.push(median);
    }

    // Take last N kernel widths equal to last computable kernel width
    trace!(
        "Taking last {} distances equal to last window median {}",
        half_smoothing_len + 1,
        median
    );
    for _ in 0..=half_smoothing_len {
        kernel_bandwidth.push(median);
    }

    // Emit final output
    debug!("Final density kernel bandwidths: {kernel_bandwidth:?}");
    kernel_bandwidth.into_boxed_slice()
}

/// A "set" of values that allows duplicates and provides fast min/max queries
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OrderedMultiset<T> {
    set: BTreeMap<T, NonZeroUsize>,
    len: usize,
}
//
impl<T: Copy + Ord> OrderedMultiset<T> {
    /// Basic constructor
    pub fn new() -> Self {
        Self {
            set: BTreeMap::new(),
            len: 0,
        }
    }

    /// Truth that this set contains no elements
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of values currently present in the set
    pub fn len(&self) -> usize {
        self.len
    }

    /// Minimal value present in the set
    pub fn min(&self) -> Option<T> {
        self.set.first_key_value().map(|(&k, _v)| k)
    }

    /// Maximal value present in the set
    pub fn max(&self) -> Option<T> {
        self.set.last_key_value().map(|(&k, _v)| k)
    }

    /// Insert a value, tell how many identical values were present before
    pub fn insert(&mut self, value: T) -> usize {
        self.insert_several(value, NonZeroUsize::new(1).unwrap())
    }

    /// Insert N copies of a value, tell how many were present before
    pub fn insert_several(&mut self, value: T, count: NonZeroUsize) -> usize {
        let result = match self.set.entry(value) {
            Entry::Vacant(v) => {
                v.insert(count);
                0
            }
            Entry::Occupied(mut o) => {
                let old_count = *o.get();
                *o.get_mut() = old_count
                    .checked_add(count.get())
                    .expect("No overflow please");
                old_count.get()
            }
        };
        self.len += count.get();
        result
    }

    /// Attempt to remove a value from the set, on success tell how many values
    /// were present before
    pub fn remove(&mut self, value: T) -> Result<NonZeroUsize> {
        let result = match self.set.entry(value) {
            Entry::Vacant(_) => bail!("Attempted to remove a nonexistent value"),
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
                Ok(old_value)
            }
        };
        self.len -= 1;
        result
    }

    /// Remove all copies of the smallest value from the set
    pub fn remove_all_min(&mut self) -> Option<(T, NonZeroUsize)> {
        self.set
            .pop_first()
            .inspect(|(_value, count)| self.len -= count.get())
    }

    /// Remove all copies of the largest value from the set
    pub fn remove_all_max(&mut self) -> Option<(T, NonZeroUsize)> {
        self.set
            .pop_last()
            .inspect(|(_value, count)| self.len -= count.get())
    }

    /// Clear this multiset
    pub fn clear(&mut self) {
        self.set.clear();
        self.len = 0;
    }
}
//
impl<T: Copy + Ord> FromIterator<T> for OrderedMultiset<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut result = Self::new();
        for value in iter {
            result.insert(value);
        }
        result
    }
}
