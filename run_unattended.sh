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
# perf and classic utilities like bash, grep and lscpu. Patches to extend
# hardware or OS support are welcome!

# Remove a few bash gotchas that I don't need
set -euo pipefail

# Remove remains of previous runs to avoid confusion
cargo clean
rm -f perf.data*

# First, qualitatively check if subnormals seem to be a problem at all
cargo bench --features=check
echo '=== If no operation is slowed down by subnormals, you can stop here ==='

# Benchmark all supported data types in an optimized codegen configuration
#
# On hardware like x86 that supports multiple SIMD vector widths, it is
# beneficial to only enable the minimum hardware target-features required for
# each vector width because it reduces the strength of the optimization barriers
# that we need to apply to prevent autovectorization, and thus the negative
# impact of such barriers on the quality of generated code.
function bench_each_type() {
    function rename_perf() {
        if [[ -e perf.data ]]; then
            mv perf.data perf.data.$1
        fi
    }
    if [[ $(lscpu | grep x86) ]]; then
        if [[ $(lscpu | grep avx512f ]]; then
            RUSTFLAGS='-C target-features=+avx512f' $* --benches=f32x16,f64x08
            rename_perf opt.avx512
        fi
        if [[ $(lscpu | grep avx) ]]; then
            RUSTFLAGS='-C target-features=+avx' $* --benches=f32x08,f64x04
            rename_perf opt.avx
        fi
        if [[ $(lscpu | grep sse2) ]]; then
            RUSTFLAGS='-C target-features=+sse2' $* --benches=f32x04,f64x02
            rename_perf opt.sse2
        fi
        RUSTFLAGS='' $* --benches=f32,f64
        rename_perf opt.scalar
    else
        if [[ -v WARNED_ABOUT_TARGET_FEATURES ]]; then
            echo '--- WARNING: Unknown hardware architecture, target-features may be suboptimal ---'
            export WARNED_ABOUT_TARGET_FEATURES=1
        fi
        $*
        rename_perf native
    fi
}

# Measure perf data for codegen quality assessment
function check_codegen() {
    cargo build --benches --features=codegen
    PERF_FLAGS=''
    if [[ $(lscpu | grep AMD) ]]
        # perf record without -a is not reliably available on AMD chips
        PERF_FLAGS="${PERF_FLAGS} -a"
    fi
    perf record ${PERF_FLAGS} -- cargo bench --features=codegen -- --profile-time=1
}
bench_each_type check_codegen
msg_native_codegen_ready='=== You can now run perf report --no-source -i perf.data.opt.xyz to check optimized codegen ==='
if [[ -e perf.data.native ]]; then
    echo "${msg_native_codegen_ready}"
fi

# Quantitatively measure the overhead of subnormals
bench_each_type cargo bench --features=measure
echo '=== You can now check the Criterion report in target/criterion/report ==='

# Assess codegen quality in -C target-cpu=native mode, if not done already
if [[ ! -e perf.data.native ]]; then
    check_codegen
    mv perf.data perf.data.native
    echo "${msg_native_codegen_ready}"
fi

# Given enough time, make sure all allowed benchmark configurations do run
bench_each_type cargo bench --all-features
echo '=== All done, please check out target/criterion and perf report --no-source perf.data.xyz ==='
