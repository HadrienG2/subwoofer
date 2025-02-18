use anyhow::{bail, Context, Result};
use criterion::Throughput;
use criterion_cbor::{DataDirectory, Search};
use log::{debug, info, trace};
use ordered_float::NotNan;
use std::{
    cmp::Ordering,
    collections::{btree_map::Entry, BTreeMap, HashMap, VecDeque},
    num::NonZeroUsize,
    ops::RangeInclusive,
    path::Path,
};

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
        let time_per_elem = data
            .avg_values
            .iter()
            .map(|avg| {
                NotNan::new(avg * throughput_norm).context("No NaN expected in timing measurements")
            })
            .collect::<Result<Box<[_]>>>()?;

        // Compute a kernel density estimate
        let kde = KernelDensityEstimator::new(time_per_elem);
        let kde_range = kde.suggested_range();
        dbg!(&kde_range);
        let kde_samples = kde.sample(
            kde_range,
            /* TODO: Make tunable, will ultimately be imposed by plot resolution */ 200,
        );
        dbg!(kde_samples);
        // TODO: Add logging to check logical correctness, then do some test plots
        // TODO: Finally, do final 2D plot that aggregates multiple subnormal
        //       proportions, by removing proportion filter above + detecting
        //       benchmark change to decide when the 2D plot changes.

        todo!()
    }
    Ok(())
}

// TODO: Extract this to a dedicated kernel_density module

/// Relative length of the sliding window used to determine the kernel density
/// estimator's kernel bandwidths via median filtering
///
/// This is a fraction of the total dataset width, so a higher value means a
/// smaller window that spans less points, leading to a kernel density estimator
/// that looks less smooth but catches narrow peaks in the probability density
/// better.
///
/// This should be set according to the smallest expected width of peaks in the
/// probability density law. When in doubt, start with high values during data
/// exploration (it reduces the odds of missing some interesting feature of the
/// data, at the expense of more false positive peaks in the probability law)
/// then shrink until you get the smoothest plot that faithfully represents the
/// data (no fake peak that doesn't seem to exist in the real timing law).
const REL_HALF_SMOOTHING_LEN: usize = 100;

/// Kernel function bandwidth as a function of the median inter-sample spacing
///
/// Higher means smoother, less spiky output with less fake modes, at the
/// expense of broadening the density estimate with respect to the actual
/// probability density distribution and hiding modes in extreme cases.
const REL_KERNEL_BANDWIDTH: f64 = 2.0;

/// Kernel density estimator for timing samples
///
/// This estimates the probability density law followed by a benchmark's timings
/// from a finite amount of timing samples. This specific implementation aims to
/// fulfill a number of desirable properties:
///
/// - Do not assume that the timing distribution is unimodal.
/// - Do not assume that modes of the timing distribution are normal.
/// - Do not assume that modes of the timing distribution are symmetrical.
/// - Produce a smooth result, with low odds of fake peaky mode that come from
///   insufficient data samples rather than the underlying probability law.
/// - Do not oversmooth to the point of hiding one mode of the timing
///   distribution by enlarging the neighboring ones too much.
/// - Have a minimal number of free parameters, so that kernel density plots can
///   be mass-produced through a largely automated process with minimal per-plot
///   manual parameter adjustment.
/// - Be reasonably cheap to compute, again, this estimator is meant to be used
///   in bulk plotting scenarios.
pub struct KernelDensityEstimator {
    sorted_samples: Box<[NotNan<f64>]>,
    kernel_bandwidths: Box<[NotNan<f64>]>,
}
//
impl KernelDensityEstimator {
    /// Set up the kernel density estimator
    pub fn new(mut samples: Box<[NotNan<f64>]>) -> Self {
        assert!(!samples.is_empty());
        samples.sort_unstable();
        let sorted_samples = samples;
        let kernel_bandwidths = kernel_bandwidth(&sorted_samples);
        Self {
            sorted_samples,
            kernel_bandwidths,
        }
    }

    /// Provide the suggested timing range for kernel density plots.
    ///
    /// With the bounded-support kernel that we currently use, this range is
    /// exact, i.e. the probability density estimate is nonzero right before
    /// the upper limit of the range and zero for any higher value.
    ///
    /// We may, in the future, switch to another kernel function that doesn't
    /// have a bounded support, like the Gaussian kernel. In that case, the
    /// definition of this function will change to indicate an upper bound above
    /// which the kernel function will do nothing but slowly decay below a
    /// negligible fraction of its maximum value (probably 1% ?).
    pub fn suggested_range(&self) -> RangeInclusive<f64> {
        let end = self
            .sorted_samples
            .iter()
            .zip(&self.kernel_bandwidths)
            .map(|(pos, width)| pos + width)
            .max()
            .expect("Should process at least one data point");
        0.0..=end.into_inner()
    }

    /// Produce `num_points` regularly spaced samples of the kernel density
    /// estimator over a certain horizontal (timing) range
    pub fn sample(&self, horizontal_range: RangeInclusive<f64>, num_points: usize) -> Box<[f64]> {
        // Check that the parameters are sensible: range is non-empty, and we
        // produce at least two points to cover both endpoints of the interval
        assert!(!horizontal_range.is_empty());
        assert!(num_points >= 2);
        let x_start = horizontal_range.start();
        let x_end = horizontal_range.end();
        let last_idx = num_points - 1;

        /// Epanechikov's parabolic kernel function
        ///
        /// This kernel function has many desirable properties: it is cheap to
        /// compute, has a bounded support (which means that we don't need to
        /// compute it everywhere) and provides optimal error convergence.
        #[derive(Clone, Copy, Debug)]
        struct Kernel {
            sample_idx: f64,
            bandwidth_as_idx: f64,
            inv_bandwidth_as_idx: f64,
        }
        //
        impl Kernel {
            // Only call with in-bound coordinates please
            fn compute(&self, idx: usize) -> f64 {
                let coord = (idx as f64 - self.sample_idx) * self.inv_bandwidth_as_idx;
                debug_assert!(coord.abs() <= 1.0);
                0.75 * self.bandwidth_as_idx * (1.0 + coord) * (1.0 - coord)
            }
        }

        // Determine which output data points are covered by the kernel of which
        // sample, and deduce when we need to start and stop taking each kernel
        // into account while iterating over output data points.
        let mut kernel_starts_before = BTreeMap::<usize, Vec<(usize, Kernel)>>::new();
        let mut kernel_ends_after = BTreeMap::<usize, Vec<usize>>::new();
        let rel_x_to_idx = last_idx as f64 / (x_end - x_start);
        for (kernel_id, (sample, bandwidth)) in self
            .sorted_samples
            .iter()
            .zip(&self.kernel_bandwidths)
            .enumerate()
        {
            // Determine output data point indices covered by this kernel
            //
            // At this stage, we are allowing fake indices (negative, etc) that
            // represent points outside of the target `horizontal_range`...
            let sample_idx = (sample - x_start) * rel_x_to_idx;
            let bandwidth_as_idx = bandwidth * rel_x_to_idx;
            let start_idx = sample_idx - bandwidth_as_idx;
            let end_idx = sample_idx + bandwidth_as_idx;
            let first_covered_idx = start_idx.ceil();
            let last_covered_idx = end_idx.floor();

            // Discard kernels which only cover out-of-range indices, then wrap
            // covered indices into the true output index range, which will
            // allow us to move to integer indices.
            if last_covered_idx < 0.0 || first_covered_idx > last_idx as f64 {
                continue;
            }
            let first_covered_idx = first_covered_idx.max(0.0) as usize;
            let last_covered_idx = last_covered_idx.min(last_idx as f64) as usize;

            // Ignore kernels that do not cover any output index
            //
            // This happens when the kernel function is so narrow that it starts
            // and ends between the same two output data points. Since we are
            // not currently performing any antialiasing on the kernel density
            // estimator, this will result in those samples' kernel not
            // contributing to the sampled output at all.
            if (first_covered_idx..=last_covered_idx).is_empty() {
                debug_assert_eq!(last_covered_idx, first_covered_idx - 1);
                continue;
            }

            // Prepare kernel function
            let kernel = Kernel {
                sample_idx: sample_idx.into_inner(),
                bandwidth_as_idx: bandwidth_as_idx.into_inner(),
                inv_bandwidth_as_idx: 1.0 / bandwidth_as_idx.into_inner(),
            };

            // Record at which output index the kernel function associated with
            // this sample becomes nonzero and thus starts to contribute to the
            // density estimate.
            //
            // We annotate this output index with both the kernel density
            // function and the associated sample index, which is later used to
            // disable the kernel function once we exit its bounded support.
            kernel_starts_before
                .entry(first_covered_idx)
                .or_default()
                .push((kernel_id, kernel));

            // If the kernel function becomes inactive before the last data
            // point is emitted domain, record where it happens so we can stop
            // computing it.
            //
            // We do not need to disable it if it becomes inactive after the
            // last generated data point has been emitted.
            if last_covered_idx < last_idx {
                kernel_ends_after
                    .entry(last_covered_idx)
                    .or_default()
                    .push(kernel_id);
            }
        }

        // Now we are ready to generate output data points
        //
        // First we track currently generated output data points and active
        // kernel functions...
        let mut output = Vec::with_capacity(num_points);
        let mut active_kernels = HashMap::<usize, Kernel>::new();

        // Then from this we define "free-wheeling" output data generation, in
        // sections of the output index domain where the set of timing samples
        // with nonzero kernel functions is constant...
        let generate_output_until =
            |output: &mut Vec<f64>, active_kernels: &HashMap<usize, Kernel>, end_idx: usize| {
                debug_assert!(output.len() <= end_idx);
                while output.len() < end_idx {
                    let next_output_idx = output.len();
                    let kernel_sum = active_kernels
                        .values()
                        .map(|kernel| kernel.compute(next_output_idx))
                        .sum();
                    output.push(kernel_sum);
                }
            };

        // Then we extend this "free-wheeling" to also cover the scenario where
        // at the end of such a run, we add new kernels to the set of active
        // kernels or we remove some of them from the set.
        let process_kernel_starts =
            |output: &mut Vec<f64>,
             active_kernels: &mut HashMap<usize, Kernel>,
             start_before: usize,
             added_kernels: &Vec<(usize, Kernel)>| {
                // Generate output until index `start_before`, exclusive since
                // `start_before` represents the index of the first output data
                // point that is covered by the added kernels...
                generate_output_until(output, active_kernels, start_before);
                // ...then add the new kernels to the kernel set so that they
                // contribute to the next data points.
                for &(kernel_idx, kernel) in added_kernels {
                    let insert_outcome = active_kernels.insert(kernel_idx, kernel);
                    debug_assert!(
                        insert_outcome.is_none(),
                        "Kernel should not be active before advertised start"
                    );
                }
            };
        let process_kernel_ends = |output: &mut Vec<f64>,
                                   active_kernels: &mut HashMap<usize, Kernel>,
                                   end_after: usize,
                                   removed_kernels: &Vec<usize>| {
            // Generate output until index `end_after`, inclusive since
            // `end_after` represents the index of the last output data point
            // that is covered by the removed kernels...
            generate_output_until(output, active_kernels, end_after + 1);
            // ...then remove the kernels from the kernel set
            for kernel_idx in removed_kernels {
                let remove_outcome = active_kernels.remove(kernel_idx);
                debug_assert!(
                    remove_outcome.is_some(),
                    "Kernel should be active before advertised end"
                );
            }
        };

        // Finally we jointly iterate over the boundaries at which kernel
        // function supports start and stop, in output index order.
        let mut kernel_starts_before = kernel_starts_before.into_iter().peekable();
        let mut kernel_ends_after = kernel_ends_after.into_iter().peekable();
        'generate_output: loop {
            match (kernel_starts_before.peek(), kernel_ends_after.peek()) {
                (Some((start_before, added_kernels)), Some((end_after, _)))
                    if start_before <= end_after =>
                {
                    // The closest boundary adds new kernel functions
                    process_kernel_starts(
                        &mut output,
                        &mut active_kernels,
                        *start_before,
                        added_kernels,
                    );
                    kernel_starts_before.next();
                }
                (Some((start_before, _)), Some((end_after, removed_kernels)))
                    if start_before > end_after =>
                {
                    // The closest boundary removes former kernel functions
                    process_kernel_ends(
                        &mut output,
                        &mut active_kernels,
                        *end_after,
                        removed_kernels,
                    );
                    kernel_ends_after.next();
                }
                (Some(_starts_before), None) => {
                    // There are only kernel addition boundaries left, so we can
                    // stop peeking and just iterate over them.
                    for (start_before, added_kernels) in kernel_starts_before {
                        process_kernel_starts(
                            &mut output,
                            &mut active_kernels,
                            start_before,
                            &added_kernels,
                        );
                    }
                    break 'generate_output;
                }
                (None, Some(_ends_after)) => {
                    // There are only kernel removal boundaries left, so we can
                    // stop peeking and just iterate over them.
                    for (end_after, removed_kernels) in kernel_ends_after {
                        process_kernel_ends(
                            &mut output,
                            &mut active_kernels,
                            end_after,
                            &removed_kernels,
                        );
                    }
                    break 'generate_output;
                }
                _ => unreachable!(),
            }
        }
        generate_output_until(&mut output, &active_kernels, num_points);
        assert_eq!(output.len(), num_points);
        output.into_boxed_slice()
    }
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
    let mut result = Vec::with_capacity(sorted_samples.len());

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
        result.push(median * REL_KERNEL_BANDWIDTH);
    }

    // Compute next kernel widths using a sliding median filter
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
        result.push(median * REL_KERNEL_BANDWIDTH);
    }

    // Take last N kernel widths equal to last computable kernel width
    trace!(
        "Taking last {} distances equal to last window median {}",
        half_smoothing_len + 1,
        median
    );
    for _ in 0..=half_smoothing_len {
        result.push(median * REL_KERNEL_BANDWIDTH);
    }

    // Emit final output
    debug!("Final density kernel bandwidths: {result:?}");
    result.into_boxed_slice()
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
