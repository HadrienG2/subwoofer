# Full measurements

To fully assess the impact of subnormal numbers of your CPU's floating-point
arithmetic performance, you will want to run a more extensive set of benchmarks
that...

- Measures the performance of more arithmetic operations
- Checks CPU register inputs in addition to memory inputs from the L1 cache
- Benchmarks SIMD floating-point types in addition to scalar ones
- Increases subnormal share resolution as needed to precisely probe the
  performance vs subnormal input share curve (e.g. measure the maximal subnormal
  number processing overhead and the subnormal input share at which this
  overhead is observed)

The `measure` [Cargo
feature](https://doc.rust-lang.org/cargo/reference/features.html) can be used to
quickly set up a benchmark configuration that has proven suitable for all the
hardware against which Subwoofer has been tested so far, at the expense of often
being overly precise and thus taking an unnecessary long time to build and run:

```bash
cargo bench --features=measure
```

If the execution time of this generic configuration is a problem, you may want
to deviate from it by reducing the number of benchmarks that is run to the
minimum needed to acquire the data that you are interested in. In the remainder
of this chapter, we will be discuss how this is done.


## Arithmetic operation set

By default, the `measure` configuration enables all supported microbenchmarks.
Depending on the results of the previous [basic check](basic-check.md), this may
be overkill and spend a large amount of time re-measuring information that you
already know with unnecessary extra precision.

Here's how to decide which benchmarks you can disable:

- If one of the `addsub`, `sqrt_positive_max`, `div_denominator_min` and
  `div_numerator_max` benchmarks was not affected by subnormals during the basic
  check, then you can disable it during the full measurement.
- If the `fma_full_max_mul` benchmark was not affected by subnormals during the
  basic check, then you can disable the `fma_addend_min`, `fma_multiplier_min`
  and `fma_full_max_mul` benchmarks during the full measurement.
- If **none** of the `mul_max` and `fma_full_max_mul` benchmarks were affected
  by subnormals, you can disable the `mul_max` benchmark during the full
  measurement.
- Finally, you cannot disable the `max` benchmark during the full measurement
  unless **all** benchmarks except for `addsub` were **unaffected** by
  subnormals.

If you are in one of those cases, you may want to stop using the catch-all
`measure` Cargo feature, and instead use finer-grained Cargo features that let
you control benchmarks on a case-by-case basis. Please check out the definition
of the `measure` and `bench_xyz` features in [Subwoofer's
Cargo.toml](https://github.com/HadrienG2/subwoofer/blob/main/Cargo.toml) to know
which set of Cargo features you should enable in this case. As an easier but
slower alternative, you may also disable those benchmarks at runtime using
`cargo bench`'s regex-based benchmark name filter.

For example, on AMD Zen 2/3 CPUs where only the performance of `MUL`, `SQRT` and
`DIV` are affected by subnormals, you could restrict the set of benchmarks at
compile time like this...

```bash
# This is correct as of 2024-12-29, but beware that the set of Cargo features
# covered by the "measure" option may evolve in future versions of Subwoofer
cargo bench --no-default-features  \
            --features=bench_max,bench_mul_max,bench_sqrt_positive_max,bench_div_numerator_max,bench_div_denominator_min,cargo_bench_support,register_data_sources,simd
```

...or, alternatively, restrict it at runtime like this:

```bash
cargo bench --features=measure -- '(max|mul_max|sqrt_positive_max|div_numerator_max|div_denominator_min)'
```


## ILP configurations

As discussed in the context of the [basic check](basic-check.md), we normally
expect CPUs to operate at peak floating-point throughput when they are fed with
code that has the highest possible amount of instruction-level parallelism. But
sometimes CPU frontend limitations get in the way and make the optimal degree of
instruction-level parallelism smaller than this.

This is why by default, we run benchmarks at both the maximum possible ILP and
half this ILP: in most cases, one of these configurations will be the fastest
possible one for your CPU, or at least very close to the performance of the
optimal ILP configuration.

If you found out during the basic check that your CPU needs an even more drastic
ILP reduction to perform at optimal throughput, you will need to enable the
`more_ilp_configurations` Cargo feature for the full benchmark that we are
discussing in this chapter. In this case, consider [sending us a bug
report](https://github.com/HadrienG2/subwoofer/issues/new/choose): if your CPU
model is sufficiently common, we may want to enable more ILP configurations by
default, as we aim for a default configuration that Just Works.

Once you've found the level of ILP that leads to peak throughput for each
benchmark, you can speed up benchmark execution by only enabling this ILP
configuration along with the `chained` latency-bound configuration. Here is an
example of applying such an ILP configuration filter when benchmarking SQRT
performance on an AMD Zen 2 CPU:

```bash
cargo bench --features=measure -- '(max/ilp08|sqrt_positive_max/ilp04)|chained'
```


## CPU register inputs

On many common CPUs, the performance of Subwoofer microbenchmarks that operate
on data from the L1 data cache is mainly limited by floating-point arithmetic
performance, rather than memory subsystem performance. However...

- This not true of all benchmarks, for example `fma_full_max` on all-normal
  inputs from the L1 cache is memory-bound on all CPUs that Subwoofer has been
  tested against so far.
- It may not be true on all CPUs because the memory operations do consume some
  CPU frontend and backend resources, and these resources could become the
  bottleneck on CPU cores that are less well-balanced than those which Subwoofer
  has been tested against so far.

For this reason, the `register_data_sources` Cargo feature, which is part of the
broader `measure` feature, enables alternate versions of the microbenchmarks
that run against inputs from CPU registers instead of inputs from the L1 cache.
This configuration avoids memory subsystem bottlenecks entirely, at the expense
of having other drawbacks:

- Because the input dataset is tiny, a sufficiently smart CPU backend could
  apply optimizations to its internal subnormal number processing logic that do
  not apply on a more realistic data scale.
- Even if the CPU does not apply such optimizations, rustc and LLVM can perform
  some of them. We apply optimization barriers to prevent them from doing so,
  but they may sometimes come at the expense of a reduction in generated code
  quality.

Because these drawbacks normally outweigh the benefits of not putting pressure
on the CPU's memory subsystem, we strongly advise against running _only_ those
versions of the benchmarks, and we suggest taking their results with a
respectable grain of salt:

- If they are a bit faster than the benchmarks that operate from the L1 cache
  (say, 20% faster), it may reflect a genuine hardware performance benefit of
  avoiding the memory subsystem, and thus a "purer" measurement of the CPU's
  subnormal number processing overhead.
- If they are >2x faster, or worse slower, you should disregard any data that
  comes out of them as suspicious by default, unless you have the time to
  carefully analyze the generated code and runtime CPU microarchitecture
  behavior to prove this default hypothesis wrong.
- In general, if you have any doubt, your default assumption should be that
  benchmarks that operate from the L1 cache are "more right" than those that
  operate from CPU registers, until proven otherwise.

If you want to disable register inputs to speed up the benchmark's build and
execution, then the most efficient way will be to refrain from enabling
`register_data_sources` Cargo feature.

Sadly, Cargo features cannot be disabled, only enabled, so the only way to
enable a subset of the `measure` feature pack is to look up its definition in
Cargo.toml and only enable the subset that you want, as discussed above in the
context of controlling the set of benchmarked operations.

An easier but slower alternative is to only disable the execution of benchmarks
at runtime using `cargo bench`'s regex filter, like this:

```bash
cargo bench --features=measure -- 'L1cache'
```


## More memory data sources

By default, we only measure the performance of floating-point operations on
memory inputs that fit in the L1 data cache, because that's where the
performance impact of subnormal numbers is expected to be the highest. As we
access increasingly remote memory inputs, the CPU is expected to spend less time
crunching numbers, and more time waiting for the memory subsystem, resulting in
a reduction of the relative subnormal number processing overhead.

This is, however, only 100% expected when processing data at the maximal SIMD
vector width supported by your CPU architecture. When data is accessed in
smaller chunks, the CPU's spatial prefetcher gets more time to hide the latency
of remote memory accesses by predicting future memory accesses before the CPU
has even requested them, and as a result performance for e.g. scalar data will
often be similar for all layers of the memory hierarchy.

If you are interesed in studying those kind of effects, consider adding
`more_memory_data_sources` to the set of Cargo features that you are enabling.
This feature is never enabled by default because it falls a bit outside of the
scope of what Subwoofer normally aims to measure (namely the impact of subnormal
numbers on the performance of floating-point arithmetic).


## SIMD data types

Depending on how the CPU's subnormals fallback is implemented, its performance
may or may not depend on the floating-point data type that is being manipulated.
In particular, it may be faster or slower for code that operates on SIMD
vectors of numbers, rather than individual "scalar" numbers.

To account for this, the `simd` Cargo feature, which is part of the broader
`measure` feature, lets you test the performance of all supported SIMD vector
types. If you are trying to speed up benchmark runs, we advise narrowing this
down to just the widest supported SIMD vector types and scalar data, as it is
unlikely that intermediate vector sizes will behave very differently. This can
be done using `cargo bench`'s regex filtering feature:

```bash
# Suitable for an x86 CPU whose native vector width is 256-bit AVX
cargo bench --features=measure -- 'f(32|64)(x08)?/'
```

Another way in which SIMD affects our benchmarks is that we need to apply
optimization barriers to prevent the compiler from auto-vectorizing our scalar
benchmarks into SIMD code, or our narrow SIMD vector benchmarks into wider SIMD
code. Unfortunately, these optimization barriers may come at the expense of a
reduction in generated code quality, so it is best to avoid them.

To do this, you will need to compile each type-specific benchmark with the
narrowest `target-feature` set that is...

1. Needed to process this particular scalar/SIMD floating-point type
2. Legal on the target CPU architecture

The provided [`run_attended.sh`
script](https://github.com/HadrienG2/subwoofer/blob/main/run_unattended.sh)
applies this approach to optimize codegen on x86 CPUs.


## Subnormal frequency resolution

The last tunable of the Subwoofer microbenchmark suite is the set of
`subnormal_freq_resolution_1inN` cargo features. It controls the number of
subnormal occurence frequencies that are probed, and thus the horizontal
resolution of the subnormal overhead vs occurence frequency graph in the
performance report that each benchmark will generate at the end.

Like many other configurable Subwoofer settings, this is a tradeoff between
benchmark execution time and output precision: the higher the frequency
resolution, the longer the benchmarks will take to run, but the more precise the
output data will be in the end.

Unless you have a good reason to do so, we would advise using a frequency
resolution that is sufficient to precisely measure the horizontal and vertical
position of the peak of maximum overhead on your hardware, as well as the
general shape of the curves on either side of this peak (linear,
exponential-like, or logarithm-like?).

Alas, the optimal resolution is hardware-dependent, and can only be found
through slow experimentation. But the default setting enabled by the `measure`
Cargo feature is known to be good enough on all CPUs where Subwoofer has been
tested so far. We'd like to keep it that way, so if you find a CPU where the
default is not precise enough, please consider [sending us a bug
report](https://github.com/HadrienG2/subwoofer/issues/new/choose).


## Maximizing coverage

So far, this chapter has focused on techniques to reduce the execution time of
Subwoofer to a minimum, while still measuring the most important data. However,
there are situations where benchmark execution time is not as much of a problem,
and the largest concern is instead to make sure all important data has been
acquired. This is for example the case when we are debugging a new benchmark, or
when benchmarking hardware which you only have temporary access to.

In this case, the simplest option is to run benchmarks with the `--all-features`
Cargo option, which will instruct Subwoofer to measure as much data as
possible...

```bash
cargo bench --all-features
```

...and then make sure that you remember to copy the contents of the
`target/criterion` directory before you lose access to the hardware or
accidentally cause a `cargo clean` disaster. The price to pay is that this
configuration is extremely slow, and will take _days_ to run to completion.

A better latency vs execution time tradeoff can be achieved by using the
provided [`run_attended.sh`
script](https://github.com/HadrienG2/subwoofer/blob/main/run_unattended.sh). It
runs multiple benchmark passes of increasing precision/execution time so that
you get basic results early, and full results eventually, at the expense of a
negligibly small increase in overall execution time with respect to
`--all-features`.