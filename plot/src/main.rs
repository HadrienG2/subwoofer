mod kernel_density_estimator;

use anyhow::{bail, Context, Result};
use criterion::Throughput;
use criterion_cbor::{DataDirectory, Search};
use kernel_density_estimator::KernelDensityEstimator;
use log::{debug, info};
use ordered_float::NotNan;
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
        let kde = KernelDensityEstimator::new(time_per_elem);
        let kde_samples = kde.sample(
            kde.suggested_range(),
            /* TODO: Make tunable, will ultimately be imposed by plot resolution */ 1080,
        );
        dbg!(kde_samples);
        // TODO: Add logging to check logical correctness, then do a plot that shows the samples and KDE
        // TODO: Finally, do final 2D plot that aggregates multiple subnormal
        //       proportions, by removing proportion filter above + detecting
        //       benchmark change to decide when the 2D plot changes.

        todo!()
    }
    Ok(())
}
