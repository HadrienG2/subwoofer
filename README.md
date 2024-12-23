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
microarchitectures, so given a week or two of interactive access to an ARM or
RISC-V machine with support for perf profiling, I think I should be able to
validate it for those ISAs too.


## Setting up your machine

### Build requirements

You are going to need `rustup` and the `libhwloc` C library along with the
associated pkg-config file. The latter should either be installed system-wide or
be reachable via your `PKG_CONFIG_PATH`.

On Unices like Linux and macOS, you can install `libhwloc` and the tooling
needed to link it with Rust code by running the following commands...

- **macOS:**
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"  \
  && brew install hwloc pkgconf
  ```
- **Debian/Ubuntu/Mint:**
  ```bash
  sudo apt-get update  \
  && sudo apt-get install build-essential libhwloc-dev libudev-dev pkg-config
  ```
- **Fedora:**
  ```bash
  sudo dnf makecache --refresh   \
  && sudo dnf group install c-development   \
  && sudo dnf install hwloc-devel libudev-devel pkg-config
  ```
- **RHEL/Alma/Rocky:**
  ```bash
  sudo dnf makecache --refresh  \
  && sudo dnf groupinstall "Devlopment tools"  \
  && sudo dnf install epel-release  \
  && sudo /usr/bin/crb enable  \
  && sudo dnf makecache --refresh  \
  && sudo dnf install hwloc-devel libudev-devel pkg-config
  ```
- **Arch:**
  ```bash
  sudo pacman -Syu base-devel libhwloc pkg-config
  ```
- **openSUSE:**
  ```bash
  sudo zypper ref  \
  && sudo zypper in -t pattern devel_C_C++  \
  && sudo zypper in hwloc-devel libudev-devel pkg-config
  ```

...then you can install `rustup` and bring it into your current shell's
environment with those commands:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain none  \
&& . "$HOME/.cargo/env"
```

The required nightly Rust toolchain will then be installed automatically on the
first run of one of the `cargo bench` commands discussed below.

On Windows, I would recommend using [Windows Subsystem for
Linux](https://learn.microsoft.com/en-us/windows/wsl/install) (aka WSL) and
following the instructions for Ubuntu above because...

- WSL offers a much better software developer user experience than native
  Windows development.
- Contrary to what you may think, the underlying Linux virtual machine will not
  get in the way of precise CPU microbenchmarking due to the magic of 
  [VT-x](https://fr.wikipedia.org/wiki/Intel_VT)/[AMD-V](https://fr.wikipedia.org/wiki/Advanced_Micro_Devices#Pacifica/AMD-V).

...but if you really want a native Windows development environment, please take
inspiration from [these setup
instructions](https://numerical-rust-cpu-d1379d.pages.math.cnrs.fr/setup/windows.html)
that I wrote for my Rust numerical computing course, which will tell you how to
set up everything needed for this benchmark plus HDF5 (you can leave out the
latter if you care about minimalism).

### Improving accuracy/reproducibility

The benchmarks are single-threaded and use robust statistics, so they should
tolerate mild background OS load like code editing or basic web browsing at a
minimal accuracy cost. But if you can afford it, running them on a quiet machine
will yield more reliable and reproducible results, among other things due to the
lack of CPU frequency scaling caused by other CPU cores intermittently waking
up.

If you have more time to set things up The Right Way (tm), other classic CPU
microbenchmarking tips apply:

- Laptops should be plugged in to an electrical outlet, as their CPU performance
  tends to vary wildly over time when operating on battery power.
- OS and vendor performance/powersaving tunables should be set up for maximal
  performance, unless you are truly interested in benchmarking the powersaving
  algorithm of your specific machine (knowing that this algorithm's detail can
  vary depending on hardware, operating system, versions of CPU microcode and
  the various system software that can adjusts CPU ACPI states...).
- If maximal output reproducibility is desired, you should also disable "turbo"
  frequency scaling and set your CPU to constantly operate at its nominal
  frequency. This can be done using tools like `intel_pstate`. In addition to
  making results more stable, this will let you more easily convert the
  time-based measurements you get out of the benchmark into cycle-based
  measurements. But the price to pay is that this configuration will also reduce
  the real-world applicability of your measurements, because most people
  actually run numerical code with CPU frequency scaling enabled...


## Running the benchmark

Because many factors could potentially affect the performance of subnormal
arithmetic, the benchmarks operates in a huge combinatorial space. To keep the
compilation and run time in check, the set of probed configurations is small by
default, and can be controlled via cargo features.

You can check the Cargo.toml file for a full description of all available cargo
features, but the following is a quick guide to the execution configurations
that you will typically wish for in the initial exploration step.

---

First of all, you can quickly check if your CPU performance seems to degrade in
the presence of subnormals, and if so which arithmetic operations are affected,
by running `cargo bench` in the default (rather minimal) configuration, which
will run to completion in less than an hour.

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
more extensive configuration, that will take a lot more time to build and run:

```bash
cargo bench --features measure
```

It is a good idea to use criterion's regex-filtering mechanism to exclude some
benchmarks from this longer benchmark run. For example, if you have previously
observed that the performance impact of subnormals is the same for f32 and f64,
you may want to only run the benchmarks that take f32 inputs like this:

```bash
cargo bench --features measure -- f32
```

You can also remove some operations that are unaffected by subnormals from the
benchmarked set using the same technique, but be sure to check out the
"Analyzing the output" section below first: you may actually need to measure the
performance of some unaffected operations in order to precisely analyze the
performance of affected operations later on.

---

If you only have temporary access to the target hardware, you can alternatively
go for the opposite approach of measuring everything that can possibly be
measured, to accumulate as much data as possible for later analysis. This can be
done, at the expense of an enormous increase of compilation and execution time
(the benchmark will take several days to run), by enabling all the cargo
features at once:

```bash
cargo bench --all-features
```

If you need to go down that route, do not forget to make a proper backup of the
`target/criterion` directory as soon as possible, before you lose access to the
hardware or run `cargo clean` without thinking!


## Naming convention

Benchmarks names folow a `type/op/ilp/source/%subnormals` structure where...

- `type` is the type of data that is being operated over. Depending on how the
  CPU's subnormal fallback part is implemented, subnormal performance might
  differ for single vs double precision, and for scalar vs SIMD operations.
- `op` is the operation that is being benchmarked. Note that in the general
  case, we are not testing only the hardware operation of interest, but the
  combination of this operation with some cheap corrective actions that resets
  the accumulator to a normal state whenever it becomes subnormal. See the
  source code and the "Anayzing the output" section below for more details.
- `ilp` is the degree of instruction-level parallelism that is present in the
  benchmark. "chained" corresponds to the ILP=1 special case, which is always
  latency-bound. Higher ILP should increase execution performance until the code
  becomes throughput bound and saturates superscalar CPU backend resources, but
  the highest ILP configurations may run into microarchitectural limits (e.g.
  CPU op cache/loop buffer trashing) that degrade runtime performance instead.
    * If you observe that the highest ILP configuration is slower than the
      next-highest configuration, I advise re-running the benchmark with
      `more_ilp_configurations` added to the set of Cargo features, in order to
      make sure that your benchmark runs do cover the fastest, most
      throughput-bound ILP configuration. But this will increase execution time.
- `source` indicates where the input data comes from. As the CPU accesses
  increasingly remote input data sources, the relative impact of subnormal
  operations should go down, because the CPU will end up spending most of its
  time waiting for inputs, not computing floating-point operations.
    * By default, we only cover the data sources where the impact of subnormals
      is expected to be the highest. If you want to measure how the impact of
      subnormals goes down when the code becomes more memory-bound, you can add
      `more_data_sources` to the set of Cargo features. But this will increase
      execution time.
- `%subnormals` indicates what proportion of subnormals is present in the input.
    * The proposed `measure` benchmark configuration tests subnormal shares with
      enough percentage resolution to precisely probe the overhead curve on all
      CPUs tested so far. Your CPU model may have an overhead curve that can be
      precisely probed with less percentage resolution (leading to faster
      benchmark runs) or that requires more percentage resolution for precise
      analysis (at the expense of slower runs). In that case you may want to
      look into the various available `subnormal_freq_resolution_1inN` Cargo
      features.

The presence of leading zeros in numbers within the benchmark name may slightly 
confuse you. This is needed to ensure that the entries of criterion reports like
`target/criterion/report/index.html` are sorted correctly, because criterion
sadly does not yet use a name sorting algorithm that handles multi-digit numbers
correctly at the time of writing...


## Analyzing the output

To study the impact of subnormals on latency-bound code, look into benchmark
timings where `ilp` is "chained". These benchmarks are made of a single long
dependency chain that does not allow the CPU parallelize execution over
superscalar units, so their execution time is just the latency of a single
operation multipled by the number of operations.

Unfortunately, there is one exception to this general rule, which is the
`sqrt_positive_addsub` benchmark. This benchmark does not feature an SQRT ->
SQRT -> SQRT... dependency chain, as doing that while guaranteeing continued
subnormal input is difficult. Therefore, this benchmark cannot currently be used
to measure SQRT latency, and its output in `chained` mode should be ignored for
now.

To study the impact of subnormals on throughput-bound code, look into benchmark
timings at the `ilp` where the measured performance is highest for the operation
of interest. This should be done in the input configuration where there is no
subnormals, because given a sufficiently inefficient hardware subnormal
fallback, operations on subnormal numbers can become latency-bound even though
operations on normal numbers are throughput-bound.

For each benchmark, Criterion is configured to compute the throughput in
operations/second. For the next analysis steps, you are going to need the
average operation duration instead. To get it, simply invert the
criterion-computed throughput.

---

Once you have the average operation duration for latency-bound and
throughput-bound benchmarks, you can estimate the underlying hardware
operation's latency/throughput using the following calculations:

* The `addsub` benchmark directly provides you with the average of the
  latencies/throughputs of ADD and SUB with a possibly subnormal operand. Those
  operations have the same latency and throughput on all hardware that I know
  of, so I did not bother creating benchmarks that measures them separately.
* By subtracting the duration of `addsub` in its maximal throughput 
  configuration from the duration of `sqrt_positive_addsub` in the same
  configuration, you can estimate the throughput of SQRT with a possibly
  subnormal operand.
* The `average` benchmark provides the latency/throughput of loading an input, 
  adding an in-register value to it, and multiplying the result by another
  in-register constant. This operation does not directly translates to a
  hardware figure or merit, but we need to benchmark it in order to interprete
  the next benchmarks which include it.
* By subtracting the duration of `average` in one configuration from the
  duration of another benchmark in the same configuration, you can estimate the
  latency/throughput of...
    - `mul_average`: MUL with a possibly subnormal operand
    - `fma_multiplier_average`: FMA with a possibly subnormal multiplier
    - `fma_addend_average`: FMA with a possibly subnormal addend
    - `fma_full_average`: FMA with possibly subnormal multiplier, addend and
      result
        * The frequency at which all inputs and the output are subnormal is the
          square of the frequency at which an individual input is subnormal.

---

The highest normal arithmetic performance / subnormals impact will usually be
observed for one of the two fastest data sources, "L1cache" and "Nregisters":

- "L1cache" behaves in a standard way by loading a respectably-sized set of data
  inputs from the L1 CPU cache. But these memory accesses, which are unrelated
  to the FP arithmetic operation that we are trying to measure, may bias the
  measurement a bit, especially if the benchmark somehow manages to saturate the
  CPU's address generation units or to be bottlenecked by the L1 cache's latency
  or bandwidth (these are all thankfully rare bottlenecks on modern
  high-performance CPU microarchitectures).
- To work around this, "Nregisters" uses a tiny set of data inputs that can stay
  resident in CPU registers for the entire duration of the benchmark. This is
  the fastest possible source of data, unlike the L1 cache it should not ever
  become a bottleneck. But there is a price to pay for avoiding the L1 cache
  like this:
    * To constantly operate on the same tiny amount of data without compiler
      over-optimization, we need to apply more aggressive optimization barriers
      to the source Rust code. This may lead to slightly worse code generation,
      e.g. appearance of some nearly-free (optimized out by CPU frontend + op
      cache) register-to-register move instructions in the output binary.
    * Because this code pattern is less common and the amount of input data is
      tiny, there is a higher chance that some CPUs will behave weirdly when
      executing it, either by overly optimizing for it or to the contrary by
      choking on it and processing it more slowly than they would process
      equivalent memory-based code.

These data sources both have their merits depending on details of your CPU
microarchitecture, which is why both are included in the recommended `measure`
configuration. But generally speaking, "Nregisters" is more likely to behave
weirdly, and its results should thus be considered less trustworthy without
further ASM and microarchitectural analysis. So if you want to pick only one
data source to speed up benchmark compilation and execution, prefer "L1cache"
like the default `cargo bench` configuration does.

Assuming you nonetheless want to analyze both...

- If "Nregisters" is a bit faster than "L1cache" (say, 20% faster), its
  measurement might be more faithful to the impact of subnormal arithmetic on
  maximally optimized code because it is not polluted by memory loads, it
  contains only the desired arithmetic and perhaps a few ~free reg-reg moves.
- If "Nregisters" is a lot faster than "L1cache" (say, 2x faster), or even
  worse, significantly slower, you should disregard its output as suspicious by
  default unless you are ready to take the time to carefully cross-check that
  the machine code generated by rustc is the expected one, that the CPU
  micro-architecture cannot overly optimize for this code pattern/input size,
  etc.

Beyond these two data sources, you can enable other ones (L2+ cache, RAM...)
that have a higher latency and lower bandwidth using the
`more_memory_data_sources` cargo feature. These slower data sources will usually
exhibit a smaller relative impact from subnormal arithmetic because the CPU will
be spending less time processing data and more time waiting for data to arrive.
But this is only strongly expected to be true when processing data with the
largest available SIMD vector width for a given CPU architecture.

That's because as long as a benchmark consumes less bytes of data per cycle than
the smallest single-core bandwidth of all involved layers of the memory
hierarchy (which themselves are usually tuned to the CPU's widest SIMD vector
width), the streaming prefetcher of the CPU can normally compensate for higher
memory load latencies by preloading data in faster caches before the CPU core
has asked for it.

---

Finally, by comparing the subnormal overhead at different subnormal input
frequencies, you can gain insight into how your CPU implements its subnormal
fallback path:

* If the observed overhead grows ~linearly to a maximum at 100% subnormal
  occurence frequency, it suggests that the extra costs of processing subnormals
  in the CPU backend predominate.
* If the overhead reaches a maximum around 50% and decays on both sides, it
  suggests that the CPU's float processing logic starts with a subnormal/normal
  branch, whose misprediction costs dominate in this least predictable input
  configuration.
* If the overhead is maximal at lower frequencies, then abruptly drops, it
  suggests that the CPU's fallback logic for handling subnormals can also handle
  normal numbers, and the CPU manufacturer takes advantage of this to avoid the
  overhead of normal/subnormal mode switches by remaining in fallback mode as
  long as the frequency of subnormals occurence remains higher than a certain
  threshold.

By comparing results from different data types on the fastest data sources, you
can additionally check for type-dependent limitations of the CPU's subnormal
fallback: is it slower on double-precision operands? Does it "serialize" SIMD
operations into scalar operations or SIMD operations of smaller width?
