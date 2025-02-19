mod kernel_density_estimator;

use anyhow::{bail, Context, Result};
use criterion::Throughput;
use criterion_cbor::{DataDirectory, Search};
use kernel_density_estimator::KernelDensityEstimator;
use log::{debug, info};
use ordered_float::NotNan;
use plotters::prelude::*;
use std::path::Path;

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
        let plot_width = 640;
        let kde = KernelDensityEstimator::new(time_per_elem.clone());
        let kde_range = /*17.75..=17.85*/kde.suggested_range();
        let kde_samples = kde.sample(kde_range.clone(), plot_width);

        // TODO: Do a plot that shows the samples and KDE
        let root = BitMapBackend::new("test.png", (plot_width as u32, 480)).into_drawing_area();
        root.fill(&WHITE)?;
        let root = root.margin(10, 10, 10, 10);
        // After this point, we should be able to construct a chart context
        let max_y = kde_samples
            .iter()
            .max_by(|x, y| x.total_cmp(y))
            .copied()
            .unwrap();
        let mut chart = ChartBuilder::on(&root)
            // Set the caption of the chart
            .caption("This is our first plot", ("sans-serif", 40).into_font())
            // Set the size of the label region
            .x_label_area_size(20)
            .y_label_area_size(40)
            // Finally attach a coordinate on the drawing area and make a chart context
            .build_cartesian_2d(*kde_range.start()..*kde_range.end(), 0f64..max_y)?;
        // Then we can draw a mesh
        chart.configure_mesh().draw()?;
        // And we can draw something in the drawing area
        chart.draw_series(LineSeries::new(
            kde_samples.iter().enumerate().map(|(idx, y)| {
                let x = kde_range.start()
                    + (kde_range.end() - kde_range.start()) * idx as f64
                        / (kde_samples.len() - 1) as f64;
                (x, *y)
            }),
            &RED,
        ))?;
        let mut point_color = RGBAColor::from(BLUE);
        point_color.3 = 1.0 / 4.0;
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
        root.present()?;
        // TODO: Finally, do final 2D plot that aggregates multiple subnormal
        //       proportions, by removing proportion filter above + detecting
        //       benchmark change to decide when the 2D plot is drawn.

        todo!()
    }
    Ok(())
}
