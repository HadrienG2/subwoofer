# Naming convention

Benchmarks names folow a `type/op/ilp/source/%subnormals` structure where...

- `type` is the type of data that is being operated over. Depending on how the
  CPU's subnormal fallback part is implemented, subnormal performance might
  differ for single vs double precision, and for scalar vs SIMD operations.
- `op` is the operation that is being benchmarked.
    * Note that we are often not benchmarking only the hardware operation of
      interest, but rather a combination of this operation with some cheap
      corrective actions that resets the accumulator to a normal state whenever
      it becomes subnormal. We will explain how to deduce raw hardware
      arithmetic performance characteristics from these measurements later in
      this chapter.
- `ilp` is the degree of instruction-level parallelism that is present in the
  benchmark.
    * "chained" corresponds to the ILP=1 special case, which is always
      latency-bound. Higher ILP should increase execution performance until the
      code becomes throughput bound and saturates superscalar CPU backend
      resources, but the highest ILP configurations may not be optimal due to
      limitations of the CPU microarchitecture (e.g. CPU op cache/loop buffer
      trashing) or the optimization barrier that we use.
    * If you observe that the highest ILP configuration is slower than the
      next-highest configuration, I advise re-running the benchmark with
      `more_ilp_configurations` added to the set of Cargo features, in order to
      make sure that your benchmark runs do cover the fastest, most
      throughput-bound ILP configuration. This will increase execution time.
- `source` indicates where the input data comes from.
    * As the CPU accesses increasingly remote input data sources, the relative
      impact of subnormal operations is expected to decrease, because the CPU
      will end up spending more of its time waiting for inputs, rather than
      computing floating-point operations.
    * By default, we only cover the data sources where the impact of subnormals
      is expected to be the highest. If you want to measure how the impact of
      subnormals goes down when the code becomes more memory-bound, you can add
      `more_data_sources` to the set of Cargo features. But this will increase
      execution time.
- `%subnormals` indicates what percentage of subnormals is present in the input.
    * The proposed `measure` benchmark configuration has enough percentage
      resolution to precisely probe the overhead curve of all CPUs tested so
      far. But your CPU model may have a different overhead curve that can be
      precisely probed with less percentage resolution (leading to faster
      benchmark runs) or that requires more percentage resolution for precise
      analysis (at the expense of slower runs). In that case you may want to
      look into the various available `subnormal_freq_resolution_1inN` Cargo
      features.
    * If you ended up needing more data points than the current `measure`
      configuration, please consider submitting a PR that makes this the new
      default. I would like to keep the `measure` configuration precise enough
      on all known hardware, if possibly suboptimal in terms of benchmark
      execution time.

The presence of leading zeros in numbers within the benchmark name may slightly 
confuse you. This is needed to ensure that the entries of criterion reports
within `target/criterion/` are sorted correctly, because criterion sadly does
not yet use a name sorting algorithm that handles multi-digit numbers correctly
at the time of writing...
