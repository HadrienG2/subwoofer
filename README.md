# Subwoofer: Assessing the impact of subnormals on your CPU's performance

## What is this?

This is a set of microbenchmarks that lets you review how subnormal numbers
affect the performance of floating-point arithmetic on your CPU
microarchitecture.

Currently supported arithmetic includes ADD, SUB, MUL, FMA, and SQRT of positive
numbers, mainly with subnormal inputs and sometimes with subnormal outputs too.
DIV is not supported yet, mainly because I haven't figured out a minimally
invasive way to automatically reset an accumulator after division by a subnormal
number (which saturates the accumulator to +inf). Suggestions welcome!

As of writing, this benchmark has only been rigorously checked for correctness
on x86_64. But it has been designed with due consideration to other common
microarchitectures, so given a week or two interactive access to an ARM or
RISC-V machine with support for perf profiling, I think I should be able to
validate it for those ISAs too.


## Setting up your machine

The benchmarks are single-threaded and use robust statistics, so they should
tolerate mild background OS load like code editing or basic web browsing at a
minimal accuracy cost. But if you can afford it, running them on a quiet machine
will yield more reproducible results, if only due to lack of CPU frequency
scaling caused by other CPU cores intermittently waking up.

Other classic CPU microbenchmarking tips apply:

- Laptops should be plugged in to an electrical outlet, as their CPU performance
  tends to vary wildly over time when operating on battery power.
- OS and vendor performance/powersaving tunables should be set up for maximal
  performance, unless you are truly interested in benchmarking the powersaving
  algorithm of your specific machine (knowing that the exact algorithm varies
  depending on hardware, operating system, versions of CPU microcode and the
  various system software that adjusts CPU ACPI states...).
- If maximal output reproducibility is desired, you should also disable "turbo"
  frequency scaling and set your CPU to constantly operate at its nominal
  frequency using tools like `intel_pstate`. As a bonus, this lets you to
  reliably convert the time-based measurements into cycle-based measurements.
  But beware that this configuration also reduces the real-world applicability
  of your measurements.

---

You are also going to need `rustup` (see the [recommended toolchain installation
procedure from the Rust project](https://www.rust-lang.org/learn/get-started))
and the `libhwloc` C library. The latter should either be installed system-wide
or be reachable via your `PKG_CONFIG_PATH`.

On Unices like Linux and macOS, you can typically get a system-wide installation
of `libhwloc` by installing a package named `hwloc`, `hwloc-devel`,
`libhwloc-dev` or similar using your preferred package manager. Unfortunately,
the packaging of `hwloc` is not consistent across package managers: some Linux
distributions like Arch have a single package that bundles the CLI tools, the
runtime libraries, and the files needed for building dependent binaries, whereas
other Linux distributions package those bits separately.


## Running the benchmark

Because many factors could potentially affect the performance of subnormal
arithmetic, the benchmarks operates in a huge combinatorial space. To keep the
compilation and run time in check, the set of probed configurations is small by
default, and can be controlled via cargo features.

You can check the Cargo.toml file for a full description of all available cargo
features, but the following is a quick guide to the execution configurations
that you will typically wish for.

---

First of all, you can quickly check if your CPU performance seems to degrade in
the presence of subnormals, and if so which arithmetic operations are affected,
by running `cargo bench` in the default (minimal) configuration.

```bash
cargo bench
```

If for some benchmarked arithmetic operations (see "Naming convention" below)
you do not see any change in execution performance as the share of subnormal
numbers in the input varies, then it is probably not worthwhile to investigate
these operations any further: your CPU seems to be able to process them at
native speed even in the presence of subnormal inputs.

---

If on the other hand your CPU's arithmetic performance does degrade in the
presence of subnormal numbers, then in order to quantitavely assess the impact
of the CPU's subnormal fallback for those operations, you can benchmark with a
more extensive configuration that takes a lot more time to build and run:

```bash
cargo bench --features measure
```

---

To save on execution time during interactive analysis, it is a good idea to use
criterion's regex-filtering mechanism to exclude some benchmarks from a run. For
example, if you previously observed that the performance impact of subnormals
does not seem to differ between f32 and f64, you can only run the benchmarks
that take f32 inputs like this:

```bash
cargo bench --features measure -- f32
```

If on the other hand you only have temporary access to the target hardware, you
may alternatively go for the opposite approach of measuring everything that can
possibly be measured, accumulating as much information as possible for later
analysis. This can be done, at the expense of an enormous increase of
compilation time and runtime (we're talking about runtimes of multiple days), by
enabling all the cargo features:

```bash
cargo bench --all-features
```

If you go down that route, do not forget to make a backup of the
`target/criterion` directory before you lose access to the hardware!


## Naming convention

Benchmarks names folow a `type/op/ilp/source/subnormals` structure where...

- `type` is the type of data that is being operated over. Depending on how the
  CPU's subnormal fallback part is implemented, subnormal performance might
  differ in single vs double precision, and for scalar vs SIMD operations.
- `op` is the operation that is being benchmarked. Note that in the general
  case, we are not testing only the hardware operation of interest, but the
  combination of this operation with some cheap corrective actions that resets
  the accumulator to a normal state whenever it becomes subnormal (see the
  source code for detail).
- `ilp` is the degree of instruction-level parallelism that is present in the
  benchmark. "chained" corresponds to the ILP=1 special case, which is always
  latency-bound. Higher ILP should increase execution performance until the code
  becomes throughput bound and saturates superscalar CPU backend resources, but
  the highest ILP configurations may run into microarchitectural limits (e.g.
  CPU op cache/loop buffer trashing) that degrade runtime performance instead.
    * If you observe this, I advise re-running the benchmark with
      `more_ilp_configurations` added to the set of Cargo features, in order to
      make sure that your benchmark runs do cover the most throughput-bound ILP
      configuration. This will increase execution time.
- `source` indicates where the input data comes from. As the CPU accesses
  increasingly remote input data sources, the relative impact of subnormal
  operations should go down, because the CPU will end up spending most of its
  time waiting for inputs, not computing floating-point operations.
    * By default, we only cover the data sources where the impact of subnormals
      is expected to be the highest. If you want to measure how it goes down
      when the code becomes more memory-bound, then add `more_data_sources` to
      the set of Cargo features. This will increase execution time.
- `subnormals` indicates how many subnormals are present in the input.

The presence of leading zeros in numbers within the benchmark name may confuse
you. This is done to ensure that entries within the criterion report at
`target/criterion/report/index.html` are sorted correctly, as criterion sadly
does not use a name sorting algorithm that handles multi-digit numbers correctly
at the time of writing.


## Analyzing the output

To assess the impact of subnormals on arithmetic operation latency, check the
benchmarks where `ilp` is "chained". To assess their impact on operation
throughput, check benchmarks at the `ilp` when the measured performance is
highest is absence of subnormals in the input.

The operation latency/throughput in (nano)seconds is the inverse of the
throughput computed by Criterion in operations/second for the corresponding
degree of ILP. Unfortunately, I did not find a way to have Criterion directly
compute and display the time/operation in the command-line benchmark output...

You can then get from the benchmarked operation's latency/throughput to an
estimate of the underlying hardware operation's latency/throughput using the
following calculations:

* The `addsub` benchmark directly provides the average of the
  latencies/throughputs of ADD and SUB with a possibly subnormal operand. Those
  operations have the same latency and throughput on all hardware that I know
  of, so I did not bother creating benchmarks that measures each separately.
* By subtracting the latency/throughput of `addsub` from the matching figure of
  merit in the `sqrt_positive_addsub` benchmark, you get the latency/throughput
  of SQRT with a possibly subnormal operand.
* The `average` benchmark provides the latency/throughput of adding an
  in-register constant followed by multiplying by another in-register constant.
  This does not directly translates to a hardware figure or merit, but we need
  it to interprete the next benchmarks.
* By subtracting the latency/throughput of `average` from the matching figure of
  merit in the next benchmarks, you get the latency/throughput of...
    - `mul_average`: MUL by a possibly subnormal operand
    - `fma_multiplier_average`: FMA with a possibly subnormal multiplier
    - `fma_addend_average`: FMA with a possibly subnormal addend
    - `fma_full_average`: FMA with possibly subnormal multiplier, addend and
      result (the frequency at which everything is subnormal is the square of
      the frequency at which one individual input is subnormal).

The tradeoff between the "Nregisters" and "L1cache" data sources is subtle. The
former leads to a more artificial code pattern that increases the odds of CPU
microarchitecture "cheating", but the latter requires a memory load that is
unrelated to the FP arithmetic operation that we are trying to measure, and may
bias the measurement a bit. This is why both data sources are included in the
recommended `measure` configuration. If the results are only slightly different,
the "Nregisters" output should be slightly more faithful to the impact of
subnormal arithmetic on maximally optimized code. But if they are very
different, you should trust the "L1cache" measurements more unless you are sure
you know your CPU microarchitecture well enough to confidently assess that it is
not overly optimizing for in-register inputs (by e.g. always predicting internal 
normal/subnormal branches correctly).

By comparing the subnormal overhead at different subnormal input frequencies,
you can gain insight into how your CPU implements its subnormal fallback:

* If overhead is maximal at high frequencies, it suggests that the extra costs
  of processing subnormals in the CPU backend predominate.
* If overhead is maximal around 50%, it suggests that the CPU's float processing
  logic starts with a subnormal/normal branch, whose misprediction costs
  predominate in this most unpredictable case.
* If overhead is maximal at lower frequencies, then abruptly drops, it indicates
  that the CPU's fallback logic for handling subnormals can also handle normal
  numbers, and the CPU manufacturer exploited this to avoid the aforementioned
  branch predictor trashing effects by staying in fallback mode as long as the
  frequency of subnormals occurence is higher than a certain threshold.

And finally, by comparing results from different data types on the fastest data
sources, you can assess some type-dependent limitations of the CPU's subnormal
fallback: is it slower on double-precision operands? Does it serialize SIMD
operations into scalar ones?
