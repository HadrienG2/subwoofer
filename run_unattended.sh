#!/bin/bash

# Optimized unattended execution configuration.
#
# This starts with the configuration that runs the quickest, then gradually
# proceeds with longer-running tests that produce more detailed output. This
# ensures, at the expense of a relatively small increase in overall running
# time, that you can get maximally useful data for a given amount of benchmark
# execution time during unattended execution.
#
# This was only tested on Linux and with x86 CPUs, and assumes availability of
# cargo criterion, perf and classic utilities like bash, grep and lscpu. Patches
# to extend hardware or OS support are welcome!

### START FROM A CLEAN SLATE ###

# Remove a few bash gotchas that I don't need
set -euo pipefail

# Remove remains of previous runs to avoid confusion
cargo clean
rm -f perf.data*

### QUICK CHECK FOR SUBNORMAL ISSUES ###

# First, qualitatively check if subnormals seem to be a problem at all
cargo criterion --features=check
echo '=== If no operation is slowed down by subnormals, you can stop here ==='

### OPTIMIZED CODEGEN, CODEGEN CHECK ###

# Benchmark all supported data types in an optimized codegen configuration
#
# On hardware like x86 that supports multiple SIMD vector widths, it is
# beneficial to only enable the minimum hardware target-features required for
# each vector width, because it reduces the strength of the optimization
# barriers that we need to apply to prevent autovectorization, and thus the
# negative impact of such barriers on the quality of generated code.
#
# Parametrized by a command that behaves like `cargo bench` and accepts cargo
# build/bench arguments, but may also produce perf.data files.
function bench_each_type() {
    function rename_perf() {
        if [[ -e perf.data ]]; then
            mv perf.data perf.data.$1
        fi
    }
    if [[ $(lscpu | grep x86) ]]; then
        FMA_FLAG='-C target-feature='
        if [[ $(lscpu | grep fma) ]]; then
            FMA_FLAG="${FMA_FLAG}+fma"
        else
            FMA_FLAG="${FMA_FLAG}-fma"
        fi
        if [[ $(lscpu | grep avx512vl) ]]; then
            # avx512vl is not really needed, but we did not implement support
            # for type-dependent register counts yet as it is not needed on any
            # real-world CPU, so we need avx512vl for the benchmark to allow
            # itself to use all available 32 registers.
            RUSTFLAGS="${FMA_FLAG},+avx512f,+avx512vl" $* --bench=f32x16 --bench=f64x08
            rename_perf opt.avx512
        fi
        if [[ $(lscpu | grep avx) ]]; then
            RUSTFLAGS="${FMA_FLAG},+avx" $* --bench=f32x08 --bench=f64x04
            rename_perf opt.avx
        fi
        # Should not activate FMA for <= SSE, as it also activates AVX and thus
        # degrades codegen.
        if [[ $(lscpu | grep sse2) ]]; then
            RUSTFLAGS='-C target-feature=+sse2' $* --bench=f32x04 --bench=f64x02
            rename_perf opt.sse2
        fi
        RUSTFLAGS='' $* --bench=f32 --bench=f64
        rename_perf opt.scalar
    else
        if [[ -v WARNED_ABOUT_TARGET_FEATURES ]]; then
            echo '--- WARNING: Unknown hardware architecture, target-features may be suboptimal ---'
            WARNED_ABOUT_TARGET_FEATURES=1
        fi
        $* --benches
        rename_perf native
    fi
}

# Decide which flags must be applied to perf record on this CPU
PERF_FLAGS=''
if [[ $(lscpu | grep AMD) ]]; then
    # perf record without -a is not reliably available on AMD chips...
    PERF_FLAGS="${PERF_FLAGS} -a"
fi

# Measure perf data for codegen quality assessment
#
# Must be parametrized with cargo's bench selection build arguments, either
# `--benches` to build all benchmarks or a sequence of `--bench=x --bench=y ...`
# to only build a specific set of benchmarks.
function check_codegen() {
    cargo build --profile=bench --features=codegen $*
    perf record ${PERF_FLAGS} -- cargo criterion --features=codegen $* -- --profile-time=1
}
bench_each_type check_codegen

# Tell the user that codegen quality measurements are now available
MSG_NATIVE_CODEGEN_READY='=== You can now run "perf report --no-source -i perf.data.native" to check default (-C target-cpu=native) codegen ==='
if [[ -e perf.data.native ]]; then
    echo "${MSG_NATIVE_CODEGEN_READY}"
else
    echo '=== You can now run "perf report --no-source -i perf.data.opt.xyz" to check optimized codegen ==='
fi

### QUANTITATIVE SUBNORMALS IMPACT ASSESSMENT ###

# Measure the overhead of subnormals with enough precision for all known CPUs
bench_each_type cargo criterion --features=measure
echo '=== You can now check the Criterion report in target/criterion/report ==='

### NATIVE CODEGEN CHECK ###

# Assess codegen quality in -C target-cpu=native mode, if not done already
if [[ ! -e perf.data.native ]]; then
    check_codegen --benches
    mv perf.data perf.data.native
    echo "${MSG_NATIVE_CODEGEN_READY}"
fi

### RUN REMAINING OPTIONAL BENCHMARKS IF GIVEN ENOUGH TIME ###

# Make sure all allowed benchmark configurations could eventually run
bench_each_type cargo criterion --all-features
echo '=== All done, please check out target/criterion and perf report --no-source perf.data.xyz ==='
