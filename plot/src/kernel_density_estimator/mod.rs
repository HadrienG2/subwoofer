//! Kernel Density Estimator for per-iteration benchmark timings

mod median_filter;

use log::{debug, trace, warn};
use median_filter::MedianFilter;
use ordered_float::NotNan;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    ops::RangeInclusive,
};

/// Relative length of the sliding window used to determine the kernel
/// bandwidths from inter-sample distances via median filtering
///
/// This is a fraction of the total number of points in the dataset, so a higher
/// value means a smaller window that spans less points, leading to a kernel
/// density estimator that looks less smooth but is more likely to handle narrow
/// peaks in the true probability density function.
///
/// This should be set according to the smallest expected width of peaks in the
/// probability density law. When in doubt, start with high values during data
/// exploration (it reduces the odds of missing some interesting feature of the
/// data, at the expense of more false positive peaks in the probability law)
/// then shrink until you get the smoothest plot that faithfully represents the
/// data (no fake peak that doesn't seem to exist in the real timing law).
const REL_HALF_SMOOTHING_LEN: usize = 20;

/// Kernel function bandwidth as a function of the median inter-sample spacing
/// in the region where this sample is located
///
/// Higher means smoother, less spiky output with less fake modes, at the
/// expense of broadening the density estimate with respect to the actual
/// probability density distribution, to the point of hiding some of the
/// probability distribution modes in extreme cases.
const REL_KERNEL_BANDWIDTH: f64 = 15.0;

/// Upper bound on the fraction of low outliers that take an unusually short
/// amount of time to run
const LOW_OUTLIER_FRACTION: f64 = 0.01;

/// Relative margin before the first non-outlier point
///
/// The plot's horizontal scale will be grown by up to this factor in an attempt
/// to include some of the low outliers into the plot and also add some visual
/// space before the first peak.
const LOW_MARGIN: f64 = 0.1;

/// Upper bound on the faction of high outliers that take an unusually large
/// amount of time to run
const HIGH_OUTLIER_FRACTION: f64 = 0.01;

/// Outlier inclusion tolerance on the high end of the plot's horizontal scale
///
/// The plot's horizontal scale will be grown by up to this factor in an attempt
/// to include more of the high outliers into the plot.
const HIGH_MARGIN: f64 = 0.1;

/// Kernel density estimator for timing samples
///
/// This estimates the probability density law followed by a benchmark's
/// per-iteration timings from a finite amount of timing samples. Our specific
/// implementation aims to fulfill a number of desirable properties:
///
/// - Do not assume that the timing distribution is unimodal.
/// - Do not assume that modes of the timing distribution are normal.
/// - Do not assume that modes of the timing distribution are symmetrical.
/// - Produce a smooth result, with low odds of fake peaks that come from
///   insufficient data samples rather than the underlying probability law.
/// - Do not oversmooth to the point of hiding one mode of the timing
///   distribution by enlarging the neighboring modes too much.
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
        let kernel_bandwidths = kernel_bandwidths(&sorted_samples);
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
    /// negligible fraction of its maximum value (probably 1%?).
    #[allow(clippy::assertions_on_constants)]
    pub fn suggested_range(&self) -> RangeInclusive<f64> {
        assert!((0.0..1.0).contains(&LOW_OUTLIER_FRACTION));
        assert!((0.0..1.0).contains(&HIGH_OUTLIER_FRACTION));
        assert!((0.0..1.0).contains(&(LOW_OUTLIER_FRACTION + HIGH_OUTLIER_FRACTION)));
        assert!(LOW_MARGIN >= 0.0);
        assert!(HIGH_MARGIN >= 0.0);

        let samples_and_bandwidths = self.sorted_samples.iter().zip(&self.kernel_bandwidths);
        let last_sample_idx = self.sorted_samples.len() - 1;
        let first_reliable_idx = (LOW_OUTLIER_FRACTION * last_sample_idx as f64).floor() as usize;
        let last_reliable_idx =
            ((1.0 - HIGH_OUTLIER_FRACTION) * last_sample_idx as f64).floor() as usize;

        let mut starts = samples_and_bandwidths
            .clone()
            .map(|(x, bw)| x - bw)
            .collect::<Box<[_]>>();
        starts.sort_unstable();
        let first_reliable_start = starts[first_reliable_idx];
        let start_dispersion = starts[last_reliable_idx] - first_reliable_start;
        let suggested_start = first_reliable_start - start_dispersion * LOW_MARGIN;

        let mut ends = samples_and_bandwidths
            .map(|(x, bw)| x + bw)
            .collect::<Box<[_]>>();
        ends.sort_unstable();
        let last_reliable_end = ends[last_reliable_idx];
        let end_dispersion = last_reliable_end - ends[first_reliable_idx];
        let suggested_end = last_reliable_end + end_dispersion * HIGH_MARGIN;

        suggested_start.into_inner()..=suggested_end.into_inner()
    }

    /// Produce `num_points` regularly spaced samples of the kernel density
    /// estimator over a certain horizontal (timing) range
    ///
    /// # Panics
    ///
    /// If the target `horizontal_range` is empty, or if `num_points` is smaller
    /// than 2, which is the minimum needed to sample at least the range edges.
    pub fn sample(&self, horizontal_range: RangeInclusive<f64>, num_points: usize) -> Box<[f64]> {
        KernelDensitySampler::new(self, horizontal_range, num_points).generate()
    }
}

/// Determine the bandwidth of the kernel associated with each timing data point
/// in the final kernel density estimator
///
/// The main tunable parameter of kernel density estimation is the width of the
/// kernel function that we associate with each individual data sample when
/// estimating the probability density function.
///
/// - If this width is taken too narrow, then the probability density estimate
///   becomes spikey (the limit case being a Dirac comb), with many fake modes
///   that do not exist in the true underlying probability law.
/// - If this width is taken too wide, then we will overestimate the width of
///   each peak in the probability law, to the point of obscuring its
///   multi-modal nature in extreme cases.
///
/// This function auto-tunes the kernel bandwidth based on a sliding median of
/// the neighboring inter-sample distances, under the premise that "spikey" is
/// what happens when the width of a sample is taken too small with respect to
/// the typical inter-sample distance in this part of the probability law.
fn kernel_bandwidths(sorted_samples: &[NotNan<f64>]) -> Box<[NotNan<f64>]> {
    // Check precondition that data must be sorted
    debug_assert!(sorted_samples.is_sorted());

    // Compute the distance between consecutive sorted samples, which will give
    // us the distances from each sample to its right neighbour (and also the
    // distance to its left neighbor if we check the previous point)
    let raw_distances = sorted_samples
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect::<Box<[_]>>();
    debug!("Will compute kernel bandwidths from raw inter-sample distances {raw_distances:#?}");

    // Set up a median filter for smoothing raw_distances
    let half_smoothing_len = raw_distances.len().div_ceil(REL_HALF_SMOOTHING_LEN);
    let full_smoothing_len = 2 * half_smoothing_len + 1;
    let (first_window, remaining_distances) = raw_distances.split_at(full_smoothing_len);
    let mut median_filter = MedianFilter::new(first_window.into());

    // Produce a smoothed version of raw_distances where...
    // - In the general case, the smoothed distance at position i is equal to
    //   the median of raw distances at positions (i-width)..=(i+width)
    // - Where this is not possible because a full window of raw data is not
    //   available (e.g. at position 0), the next or previous output of the
    //   median filter is used instead.
    let first_median = median_filter.median();
    let smoothed_distances = std::iter::repeat_n(first_median, half_smoothing_len + 1)
        .chain(
            (remaining_distances.iter().copied().map(Some))
                .chain(std::iter::repeat(None))
                .map(|distance_then_none| {
                    if let Some(raw_distance) = distance_then_none {
                        median_filter.update(raw_distance)
                    } else {
                        median_filter.median()
                    }
                }),
        )
        .take(sorted_samples.len());

    // Give each sample a bandwidth that is proportional to the smoothed
    // distance between samples around its location
    let bandwidths = smoothed_distances
        .map(|distance| distance * REL_KERNEL_BANDWIDTH)
        .collect();
    debug!("Final kernel bandwidths: {bandwidths:#?}");
    bandwidths
}

/// Harness for regularly sampling a kernel density estimator over N points in
/// a specific range of horizontal (timing) coordinates
struct KernelDensitySampler {
    /// Number of output data points that we are going to generate
    num_points: usize,

    /// Output index before which a kernel function starts contributing to the
    /// output kernel density estimate
    kernel_starts_before: BTreeMap<usize, Vec<(KernelID, KernelFunction)>>,

    /// Output index after which a kernel function stops contributing to the
    /// output kernel density estimate
    kernel_ends_after: BTreeMap<usize, Vec<KernelID>>,
}
//
impl KernelDensitySampler {
    /// Get ready to sample a kernel density estimator
    fn new(
        estimator: &KernelDensityEstimator,
        horizontal_range: RangeInclusive<f64>,
        num_points: usize,
    ) -> Self {
        // Check that the parameters are sensible: range is non-empty, and we
        // produce at least two points to cover both endpoints of the interval
        assert!(!horizontal_range.is_empty(), "Cannot sample empty domain");
        assert!(
            num_points >= 2,
            "Need at least two point to sample domain edges"
        );

        // Shorthands for manipulating horizontal range edges and transforming
        // horizontal coordinates into output data indices.
        let x_start = horizontal_range.start();
        let x_end = horizontal_range.end();
        let last_idx = num_points - 1;
        let rel_x_to_idx = last_idx as f64 / (x_end - x_start);
        debug!("Asked to sample a kernel density estimator over {num_points} points in range {horizontal_range:?} (horizontal step {})", 1.0 / rel_x_to_idx);

        // Normalization factor to be applied to each kernel function's
        // amplitude so that the integral of the kernel density estimator is
        // equal to 1 as it should be.
        let overall_norm = 1.0 / estimator.sorted_samples.len() as f64;

        // Determine which output data points are covered by the kernel
        // associated with each sample, and deduce at which output index we need
        // to start and stop taking each kernel into account while iterating
        // over output data points.
        let mut kernel_starts_before = BTreeMap::<usize, Vec<(KernelID, KernelFunction)>>::new();
        let mut kernel_ends_after = BTreeMap::<usize, Vec<KernelID>>::new();
        for (kernel_id, (sample, bandwidth)) in estimator
            .sorted_samples
            .iter()
            .zip(&estimator.kernel_bandwidths)
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
            trace!(
                "Timing sample {sample} with bandwidth {bandwidth}...\n\
                - Falls at generalized output index {sample_idx}\n\
                - Has bandwidth {bandwidth_as_idx} in output index space\n\
                - Covers generalized output index range {first_covered_idx}..={last_covered_idx}"
            );

            // Discard kernels which only cover out-of-range indices, then wrap
            // covered indices into the true output index range, which will
            // allow us to move to integer output indices.
            if last_covered_idx < 0.0 || first_covered_idx > last_idx as f64 {
                trace!(
                    "Discarding sample since it fully falls outside of the target horizontal_range"
                );
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
            // contributing to the sampled kernel density estimate at all.
            if (first_covered_idx..=last_covered_idx).is_empty() {
                trace!("Discarding sample since it does not cover any output KDE sample");
                debug_assert_eq!(last_covered_idx, first_covered_idx - 1);
                continue;
            }

            // Prepare kernel function for this sample
            let inv_bandwidth_as_idx = 1.0 / bandwidth_as_idx.into_inner();
            let kernel = KernelFunction {
                sample_idx: sample_idx.into_inner(),
                inv_bandwidth_as_idx,
                norm: 0.75 * inv_bandwidth_as_idx * overall_norm,
            };

            // Record at which output index the kernel function associated with
            // this sample becomes nonzero and thus starts to contribute to the
            // density estimate.
            //
            // We annotate this output index with both the kernel density
            // function and the associated sample index, which is later used to
            // disable the kernel function once we exit its bounded support.
            trace!("Recording that this sample starts covering output KDE samples before index {first_covered_idx} with kernel function #{kernel_id}: {kernel:#?}");
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
                trace!("...and stops covering output KDE samples after index {last_covered_idx}");
                kernel_ends_after
                    .entry(last_covered_idx)
                    .or_default()
                    .push(kernel_id);
            }
        }
        Self {
            num_points,
            kernel_starts_before,
            kernel_ends_after,
        }
    }

    /// Generate output data points
    fn generate(self) -> Box<[f64]> {
        // Prepare storage for output and active kernel functions
        let mut output = Vec::with_capacity(self.num_points);
        let mut active_kernels = HashMap::<KernelID, KernelFunction>::new();

        // From this we define "free-wheeling" output data generation, in
        // sections of the output index domain where the set of active kernel
        // functions is constant.
        let generate_output_until = |output: &mut Vec<f64>,
                                     active_kernels: &HashMap<KernelID, KernelFunction>,
                                     end_idx: usize| {
            trace!(
                "Generating data points at indices {}..{end_idx} with set of kernel functions {:?}",
                output.len(),
                active_kernels.keys().copied().collect::<BTreeSet<_>>()
            );
            debug_assert!(output.len() <= end_idx);
            if active_kernels.is_empty() {
                output.extend(std::iter::repeat_n(0.0, end_idx - output.len()));
            } else {
                while output.len() < end_idx {
                    let next_output_idx = output.len();
                    let kernel_sum = active_kernels
                        .values()
                        .map(|kernel| kernel.compute_inbounds(next_output_idx))
                        .sum();
                    output.push(kernel_sum);
                }
            }
        };

        // This is then extended to cover the full process of generating data up
        // to a boundary where the set of active kernel function changes, then
        // acknowledging this change by changing the set of active kernels.
        let process_kernel_starts =
            |output: &mut Vec<f64>,
             active_kernels: &mut HashMap<KernelID, KernelFunction>,
             start_before: usize,
             added_kernels: &Vec<(KernelID, KernelFunction)>| {
                // Generate output until index `start_before`, exclusive since
                // `start_before` represents the index of the first output data
                // point that is covered by the newly added kernels...
                generate_output_until(output, active_kernels, start_before);
                // ...then add the new kernels to the kernel set so that they
                // contribute to the next data points.
                trace!(
                    "Adding kernel functions {:?} to the active set",
                    added_kernels
                        .iter()
                        .map(|(id, _kernel)| *id)
                        .collect::<BTreeSet<_>>()
                );
                for &(kernel_idx, kernel) in added_kernels {
                    let insert_outcome = active_kernels.insert(kernel_idx, kernel);
                    debug_assert!(
                        insert_outcome.is_none(),
                        "Kernel should not be active before advertised start"
                    );
                }
            };
        let process_kernel_ends =
            |output: &mut Vec<f64>,
             active_kernels: &mut HashMap<KernelID, KernelFunction>,
             end_after: usize,
             removed_kernels: &Vec<KernelID>| {
                // Generate output until index `end_after`, inclusive since
                // `end_after` represents the index of the last output data point
                // that is covered by the kernels that are going to be removed...
                generate_output_until(output, active_kernels, end_after + 1);
                // ...then remove these kernels from the active kernel set
                trace!(
                    "Removing kernel functions {:?} from the active set",
                    removed_kernels.iter().copied().collect::<BTreeSet<_>>()
                );
                for kernel_idx in removed_kernels {
                    let remove_outcome = active_kernels.remove(kernel_idx);
                    debug_assert!(
                        remove_outcome.is_some(),
                        "Kernel should be active before advertised end"
                    );
                }
            };

        // Finally we jointly iterate over the boundaries at which kernel
        // function start and stop contributing to the output, in output index
        // order, and apply the logic defined above.
        debug!("Beginning to sample kernel density estimator");
        let mut kernel_starts_before = self.kernel_starts_before.into_iter().peekable();
        let mut kernel_ends_after = self.kernel_ends_after.into_iter().peekable();
        'generate_output: loop {
            match (kernel_starts_before.peek(), kernel_ends_after.peek()) {
                (Some((start_before, added_kernels)), Some((end_after, _)))
                    if start_before <= end_after =>
                {
                    // New kernel functions start contributing at next boundary
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
                    // Active kernel functions stop contributing at next boundary
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
        // Finally, we generate the last few data points with the kernel
        // functions that remain active until the end of the output
        generate_output_until(&mut output, &active_kernels, self.num_points);
        assert_eq!(output.len(), self.num_points);
        output.into_boxed_slice()
    }
}

/// Identifier of a kernel function, guaranteed to be unique
type KernelID = usize;

/// Epanechnikov's parabolic kernel function
///
/// We picked this kernel function because it has many good properties:
///
/// - It achieves optimal mean square error in the easy case where we are
///   estimating a Gaussian probability law
/// - It is very cheap to compute (just a few multiplies and adds)
/// - It does not feature higher-order polynomials (which can cause numerical
///   stability issues in floating-point computations)
/// - It has a smooth shape, so it is less likely than a triangular or
///   rectangular kernel to produce fake spikes in the kernel density estimate.
/// - It has a bounded support, which means that the kernel density estimator
///   computation does not need to be O(N²) or impose arbitrary cutoffs on which
///   kernels get integrated into the result
///
/// To reduce the number of computations that are performed each time the kernel
/// function is evaluated to a minimum, all kernel function parameters are
/// transformed from the space of execution times to the space of output data
/// point indices.
#[derive(Clone, Copy, Debug)]
struct KernelFunction {
    /// Position of the timing sample in the space of output indices
    sample_idx: f64,

    /// Inverse of the kernel bandwidth in the space of output indices
    inv_bandwidth_as_idx: f64,

    /// Normalization factor that includes...
    ///
    /// - The 3/4 prefactor from the Epanechnikov kernel definition
    /// - `inv_bandwidth_as_idx` to compensate for the change in kernel function
    ///   integral that is caused by broadening or shrinking it.
    /// - A 1/Nsamples factor so that the integral of the full kernel density
    ///   estimator (which is the sum of all kernel functions) is 1.0.
    norm: f64,
}
//
impl KernelFunction {
    /// Evaluate this kernel function's contribution to an output data point
    /// located at index `idx`
    fn compute_inbounds(&self, idx: usize) -> f64 {
        let u = (idx as f64 - self.sample_idx) * self.inv_bandwidth_as_idx;
        if cfg!(debug_assertions) && (u.abs() - 1.0 > 0.0) {
            warn!("Useless kernel function computation with out-of-range coordinate {u}");
        }
        let result = self.norm * (1.0 + u) * (1.0 - u);
        result.max(0.0)
    }
}
