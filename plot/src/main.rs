mod kernel_density_estimator;

use anyhow::{bail, ensure, Context, Result};
use criterion::Throughput;
use criterion_cbor::{DataDirectory, Search};
use kernel_density_estimator::KernelDensityEstimator;
use log::{debug, info};
use ordered_float::NotNan;
use plotters::prelude::*;
use std::{
    path::{Path, PathBuf},
    str::FromStr,
};

/// Plot width in pixels
const PLOT_WIDTH: u32 = 1024;

/// Supersampling antialiasing factor for plotted curves
const PLOT_ANTIALIASING: usize = 8;

/// Plot aspect ratio
const PLOT_ASPECT_RATIO: f64 = 4.0 / 3.0;

/// Plot height
const PLOT_HEIGHT: u32 = (PLOT_WIDTH as f64 / PLOT_ASPECT_RATIO) as u32;

fn main() -> Result<()> {
    // Set up logging
    env_logger::init();

    // Locate target directory
    let target_path = if let Some(path) = std::env::args().nth(1) {
        let path = PathBuf::from(path);
        ensure!(
            path.exists(),
            "Asked to operate on nonexistent target directory"
        );
        path
    } else {
        let mut workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"));
        while !workspace_root.join("Cargo.lock").exists() {
            workspace_root = workspace_root
                .parent()
                .context("Failed to locate workspace root")?;
        }
        workspace_root.join("target")
    };

    // Determine where output data should be stored
    let mut plots_root = target_path.to_owned();
    plots_root.push("plots");
    std::fs::create_dir_all(&plots_root).context("Failed to create output directory")?;

    // FIXME: Starting with just a few benchmark configurations
    let benchmark_filter = |dir: DataDirectory| {
        let dir_name = dir.dir_name();
        match dir.depth() {
            1 => {
                let Some(type_op_ilp_cache) = dir_name.strip_suffix("cache") else {
                    debug!("Ignoring benchmarks from {dir_name}: Input is not from a CPU cache");
                    return false;
                };
                let (type_op_ilp_l, cache_level) =
                    type_op_ilp_cache.split_at(type_op_ilp_cache.len() - 1);
                assert!(("1"..="9").contains(&cache_level));
                let Some(_type_op_ilp) = type_op_ilp_l.strip_suffix("_L") else {
                    unreachable!(
                        "Directory name {dir_name} doesn't end with a cache level as expected"
                    );
                };
                true
            }
            2 => {
                assert!(dir_name.ends_with('%'));
                true
            }
            _ => unreachable!(
                "Unexpected benchmark path {}",
                dir.path_from_data_root().display()
            ),
        }
    };

    // Process benchmarks
    let mut last_bench_group = None;
    let mut kde_list = Vec::new();
    let mut kde_range_end = 0.0f64;
    //
    for bench in Search::in_target_dir(target_path).find_in_paths(benchmark_filter) {
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
        let kde = KernelDensityEstimator::new(time_per_elem.clone());

        // Plot the kernel density estimator
        let bench_path = bench.path_from_data_root();
        let bench_plot_path = plots_root.join(bench_path).with_extension("png");
        plot_kde(&bench_plot_path, &time_per_elem, &kde)
            .context("Failed to plot kernel density estimator")?;

        // Determine the share of subnormals in this benchmark's dataset
        let subnormal_share = bench_path
            .file_name()
            .context("Failed to extract subnormals fraction")?
            .to_str()
            .context("Subnormals fraction should be Unicode text")?
            .strip_suffix("%")
            .context("Subnormals fraction should be a percentage")?;
        let subnormal_share =
            f64::from_str(subnormal_share).context("Failed to parse subnormal share")?;

        // Check if this benchmark belongs to the same group as previous ones
        let bench_group = bench_path
            .parent()
            .context("Expected benchmarks to have parents")?;
        let my_kde_range_end = *kde.suggested_range().end();
        if let Some(last_group) = &last_bench_group {
            if bench_group == last_group {
                // If so, update group metadata
                kde_list.push((subnormal_share, kde));
                kde_range_end = kde_range_end.max(my_kde_range_end);
            } else {
                // If not, plot KDE summary for this group...
                let group_plot_path = plots_root.join(last_group).with_extension("png");
                plot_all_kdes(&group_plot_path, kde_range_end, &kde_list[..]).context(
                    "Failed to plot all kernel density estimators from a benchmark group",
                )?;

                // ...then create a new benchmark group
                last_bench_group = Some(bench_group.to_owned());
                kde_list.clear();
                kde_list.push((subnormal_share, kde));
                kde_range_end = my_kde_range_end;
            }
        } else {
            // Also create a new benchmark group if none exists
            last_bench_group = Some(bench_group.to_owned());
            assert!(kde_list.is_empty());
            kde_list.push((subnormal_share, kde));
            kde_range_end = my_kde_range_end;
        }
    }

    // Draw last pending group plot at the end
    if let Some(last_group) = last_bench_group {
        let group_plot_path = plots_root.join(last_group).with_extension("png");
        plot_all_kdes(&group_plot_path, kde_range_end, &kde_list[..])
            .context("Failed to plot all kernel density estimators from a benchmark group")?;
    }
    Ok(())
}

/// Plot a kernel density estimator
fn plot_kde(
    plot_path: &Path,
    time_per_elem: &[NotNan<f64>],
    kde: &KernelDensityEstimator,
) -> Result<()> {
    // Create plot parent directory as needed
    if let Some(parent_dir) = plot_path.parent() {
        if !parent_dir.exists() {
            std::fs::create_dir_all(parent_dir)
                .context("Failed to create KDE plot parent directory")?;
        }
    }

    // Sample density estimator
    let kde_range = kde.suggested_range();
    let kde_samples = kde.sample(kde_range.clone(), PLOT_WIDTH as usize * PLOT_ANTIALIASING);

    // Determine vertical scale
    let max_density = kde_samples
        .iter()
        .max_by(|x, y| x.total_cmp(y))
        .copied()
        .unwrap();

    // Set up drawing area
    let root = BitMapBackend::new(&plot_path, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.margin(10, 10, 10, 10);

    // Set up chart
    let mut chart = ChartBuilder::on(&root)
        .caption("Average iteration duration probability", ("sans-serif", 35))
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(*kde_range.start()..*kde_range.end(), 0f64..max_density)?;

    // Set up axis mesh
    chart
        .configure_mesh()
        .label_style(("sans-serif", 18))
        .axis_desc_style(("sans-serif", 22))
        .x_desc("Average iteration duration (ns)")
        .y_desc("Probability density")
        .draw()?;

    // Draw smoothed probability estimate
    chart.draw_series(LineSeries::new(
        kde_samples.iter().enumerate().map(|(idx, y)| {
            let x = kde_range.start()
                + (kde_range.end() - kde_range.start()) * idx as f64
                    / (kde_samples.len() - 1) as f64;
            (x, *y)
        }),
        &RED,
    ))?;

    // Draw raw data samples
    let mut point_color = RGBAColor::from(BLUE);
    point_color.3 = 1.0 / 5.0;
    chart.draw_series(PointSeries::of_element(
        time_per_elem.iter().map(|x| (x.into_inner(), 0.0)),
        1.0,
        ShapeStyle {
            color: point_color,
            filled: true,
            stroke_width: 0,
        },
        &|coord, size, style| {
            EmptyElement::at(coord) + Rectangle::new([(-1, 0), (1, (-size * 10.0) as _)], style)
        },
    ))?;

    // Emit the final plot
    root.present()?;
    Ok(())
}

/// Plot all the kernel density estimators from a benchmark group
fn plot_all_kdes(
    plot_path: &Path,
    kde_range_end: f64,
    kde_list: &[(f64, KernelDensityEstimator)],
) -> Result<()> {
    // Create plot parent directory as needed
    if let Some(parent_dir) = plot_path.parent() {
        if !parent_dir.exists() {
            std::fs::create_dir_all(parent_dir)
                .context("Failed to create KDE plot parent directory")?;
        }
    }

    // Set up drawing area
    let root = BitMapBackend::new(&plot_path, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root.fill(&RGBColor(192, 192, 192))?;
    let root_margin = 10;
    let root = root.margin(root_margin, root_margin, root_margin, root_margin);

    // Set up chart
    let kde_range = 0.0..=kde_range_end;
    let mut chart = ChartBuilder::on(&root)
        .caption("Average iteration duration probability", ("sans-serif", 35))
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(0f64..100f64, *kde_range.start()..*kde_range.end())?;

    // Set up axis mesh
    chart
        .configure_mesh()
        .label_style(("sans-serif", 18))
        .axis_desc_style(("sans-serif", 22))
        .x_desc("Subnormal share (%)")
        .y_desc("Average iteration duration (ns)")
        .draw()?;

    // Prepare to draw data bitmap
    let plotting_area = chart.plotting_area();
    let (x_buckets, y_buckets) = plotting_area.dim_in_pixel();
    let (mut base_x, mut base_y) = plotting_area.get_base_pixel();
    base_x -= root_margin;
    base_y -= root_margin;
    let gradient = colorous::INFERNO;

    // Sample all density estimator
    let pixel_norm = 1.0 / PLOT_ANTIALIASING as f64;
    let kde_pixels = kde_list
        .iter()
        .map(|(subnormals_share, kde)| {
            let samples = kde.sample(kde_range.clone(), y_buckets as usize * PLOT_ANTIALIASING);
            let mut pixels = samples
                .chunks_exact(PLOT_ANTIALIASING)
                .map(|chunk| chunk.iter().sum::<f64>() * pixel_norm)
                .collect::<Box<[_]>>();
            let value_norm = 1.0 / pixels.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
            pixels.iter_mut().for_each(|value| *value *= value_norm);
            (subnormals_share, pixels)
        })
        .collect::<Box<[_]>>();

    // Draw smoothed probability estimate
    let x_buckets_per_subnormal_percent = x_buckets as f64 / 100.0;
    for kde_pair in kde_pixels.windows(2) {
        // For each consecutive pair of subnormal data shares...
        let (&left_subnormals, left_pixels) = &kde_pair[0];
        let (&right_subnormals, right_pixels) = &kde_pair[1];
        assert!(
            left_subnormals <= right_subnormals,
            "Subnormal fractions should be received in sorted order, but got {left_subnormals} > {right_subnormals}"
        );
        debug_assert_eq!(
            left_pixels.len(),
            right_pixels.len(),
            "All KDEs should have the same number of pixels"
        );

        // Determine which X coordinates within the plot subnormal data shares correspond to
        let first_x_bucket = (left_subnormals * x_buckets_per_subnormal_percent).round() as u32;
        let last_x_bucket = (right_subnormals * x_buckets_per_subnormal_percent).round() as u32;
        let middle_x_bucket = (first_x_bucket + last_x_bucket) / 2;

        // Proceed to draw this X region line by line
        for (y_bucket, (&left_pixel, &right_pixel)) in
            left_pixels.iter().zip(right_pixels).rev().enumerate()
        {
            for x_bucket in first_x_bucket..last_x_bucket {
                let pixel = if x_bucket < middle_x_bucket {
                    left_pixel
                } else {
                    right_pixel
                };
                let color = gradient.eval_continuous(pixel);
                root.draw_pixel(
                    (x_bucket as i32 + base_x, y_bucket as i32 + base_y),
                    &RGBColor(color.r, color.g, color.b),
                )?;
            }
        }
    }

    // Emit the final plot
    root.present()?;
    Ok(())
}
