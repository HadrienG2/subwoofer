# Basic check

Before you perform any quantitative measurements, you should first quickly check
if your CPU performance degrades in the presence of subnormal numbers. This can
be done by running `subwoofer` in the default, rather minimal configuration,
which is optimized for relatively fast execution time (<1h) at the expense of
exhaustivity.

Start by downloading the Subwoofer source code and going into it if you have not
done so already...

```bash
git clone --depth=1 https://github.com/HadrienG2/subwoofer.git
cd subwoofer
```

...then run the benchmark in its default configuration:

```bash
cargo bench
```

Here is what you should pay attention at this stage:

- If the performance of some arithmetic operations does not ever change as the
  share of subnormal inputs (percentage at the end of the benchmark name)
  varies, you _may_[^1] not need to measure the performance of this arithmetic
  operation at all.
  * If this is true of all arithmetic operations that Subwoofer measures,
    congratulations! It looks like your CPU's manufacturer did not cut any
    corner when it comes to subnormal numbers and made sure they are processes
    them at the same speed as normal floating-point numbers. In this case, it is
    likely pointless to run any other benchmark, but you can consider using the
    time you saved to send your CPU manufacturer our thanks for making the life
    of numerical computation authors less difficult.
- For each operation of interest, you should check what degree of
  instruction-level parallelism (ILP) leads to maximal performance. The
  following scenarios are expected:
  * Measured performance is maximal when ILP is maximal, as expected. This is
    the best-case scenario, you have nothing to do in this case.
  * Performance is maximal at an ILP smaller than the probed maximum. In this
    case, the max-ILP configuration is running into some CPU microarchitecture
    bottleneck, and you will want to run `cargo bench
    --features=more_ilp_configurations` to check more ILP configurations. Take
    note of the ILP that provides maximal performance in this case, and make
    sure it is not `chained`. If it is `chained`, this suggests a major codegen
    problem on our side, so you should [send us a bug
    report](https://github.com/HadrienG2/subwoofer/issues/new/choose) and
    refrain from taking any further measurement as their results will likely be
    incorrect.
- If the observed relative performance degradation is the same for `f32` and
  `f64`, then you may focus on only one of these floating-point precisions in
  the following data acquisition steps, which means that your benchmarks will
  run 2x faster.
- If the runtime performance follows a very simple degradation pattern as the
  share of subnormal inputs grows (e.g. simple affine increase), you may keep
  the number of subnormal share data points low, which will greatly speed up
  benchmark execution.

[^1]: You should read the next chapter before disabling operation benchmarks
      because many of the operations that Subwoofer measure are not elementary
      hardware arithmetic operations. This means that you need to know the
      performance of some measured operations, even if they are not affected by
      subnormals, in order to estimate the performance of other hardware
      operations that are affected by subnormals.
