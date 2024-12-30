# Latency and throughput

Modern CPUs are
[pipelined](https://en.wikipedia.org/wiki/Instruction_pipelining) and
[superscalar](https://en.wikipedia.org/wiki/Superscalar_processor), which allows
them to execute multiple instructions at the same time. This means that the
performance of a CPU instruction cannot be fully defined by a single figure of
merit. Instead, two standard figures of merit are normally used:

- **Latency** measures the amount of time that elapses from the moment where a
  CPU instruction starts executing, to the moment where the result is available
  and another instruction that depends on it can start executing. It is normally
  measured in nanoseconds or CPU clock cycles, the latter being more relevant
  when all execution bottlenecks are internal to the CPU.
- **Throughput** measures the maximal rate at which a CPU's backend can execute
  an infinite stream of instructions of a certain type, assuming all conditions
  are met (execution ports are available, inputs are ready, etc). It is normally
  given in instructions per second or per CPU clock cycle. Sometimes, people
  also provide reciprocal throughput in average CPU clock cycles per
  instruction, which is more tracherous to the reader because it looks a lot
  like a latency.

Depending on the "shape" of its machine code, a certain numerical computation
will be more limited by one of these two performance limits:

- Programs with a single long dependency chain, where each instruction depends
  on the output of the previous instruction, are normally limited by instruction 
  execution latencies
- Programs with many independent instruction streams that do not depend on each
  other are normally limited by instruction execution throughputs
- Outside of these two extreme situations, precise performance characteristics
  depend on microarchitectural trade secrets that are not well documented, and
  it is better to empirically measure the performance of code than to try to
  theoretically predict it. But we do know that observed performance will
  normally lie somewhere between the two limits of latency-bound and
  throughput-bound execution.

Subwoofer attempts to measure latency and throughput by benchmarking a varying
number of identical instruction streams where the output of each operation is
the input of the next one.

- In the `chained` configurations, there is only one instruction stream, so we
  expect latency-bound performance. The execution time for a chain of N
  operations should therefore be N times the execution latency of an individual
  operation.
- In one of the configurations of higher Instruction-Level Parallelism (ILP),
  which is normally close to the maximum ILP that is allowed by the CPU ISA,
  maximal performance will be observed. At this throughput-bound limit,
  throughput can be estimated as the number of operations that was gets
  computed, divided by the associated execution time.
    * More precisely, you should find the degree of ILP that is associated with
      maximal performance when operating on fully normal floating-point data.
      That's because code which is throughput-bound on normal inputs, may become
      latency-bound on subnormal inputs, if the CPU's subnormal fallback is so
      inefficient that it messes up the superscalar pipeline and reduces or
      fully prevents parallel instruction execution.

There is unfortunately one exception to the "`chained` is latency-bound" general
rule, which is the `sqrt_positive_max` benchmark. This benchmark does not
feature an SQRT → SQRT → SQRT... dependency chain, because performing such a
sequence of operations while guaranteeing continued subnormal input is
difficult. Therefore, this benchmark cannot currently be used to measure SQRT
latency, and its output in `chained` mode should be ignored for now.
