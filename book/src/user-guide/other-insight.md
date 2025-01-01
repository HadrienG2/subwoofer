# Other insight

## Data sources

As discussed in the data acquisition section, the suggested `measure` data
acquisition configuration exercises the hardware operation of interest not just
with memory inputs taken from the L1 cache, but also with tiny sets of inputs
that stay resident in CPU registers.

The latter configuration fully avoids memory subsystem bottlenecks, which are
rare at the L1 cache level but do exist in a few corner cases like FMAs with two
data inputs. However, it does so at the expense of being a lot less realistic
and a lot easier for compiler optimizers and CPU frontends to analyze, which
causes all sorts of problems.

As a result, you should generally consider the configuration that operates from
the L1 cache as the most reliable reference measurement, but perform a quick
comparison with the performance of the configurations that operate from
registers:

- If a configuration that operates from registers is a bit faster (say, ~20%
  faster), it is likely that the configuration with inputs from the L1 cache was
  running into a memory subsystem bottleneck, and the configuration with inputs
  from CPU registers more accurately reflects the performance of the underlying
  floating-point arithmetic operations.
- But if the difference is enormous, or the configuration that operates from
  registers is slower than the one that operates from the L1 cache, you should
  treat the results obtained against register inputs as suspicious by default,
  and disregard them until you have taken the time to apply further analysis.

Inputs from memory data sources that are more remote from the CPU than the L1
cache are generally not particularly interesting, as they are normally less
affected by the performance of floating-point arithmetic and more heavily
bottlenecked by the memory subsystem. This is why the associated performance
numbers are not measured by default.

But there is some nuance to the generic statement above (among other things
because modern CPUs have spatial prefetchers), so if you are interested in how
the performance of subnormal inputs differs when operating from a non-L1 data
source, consider trying out the `more_memory_data_sources` Cargo feature which
measures just that. This comes at the expense of, you guessed it, longer
benchmark execution times.


## Overhead vs %subnormals curve

For those arithmetic operations that are affected by subnormal inputs on a given
CPU microarchitecture, one might naively expect the associated overhead to grow
linearly as the share of subnormal numbers in the input data stream increases.

If we more rigorously spell out the underlying intuition, it is that processing
a normal number has a certain cost N, processing a subnormal number has another
cost S, and therefore the average floating-point operation cost that we
eventually measure should be a weighted mean of these two costs `x*S + (1-x)*N`
where `x` is the share of subnormal numbers in the input data stream.

This is indeed one possible hardware behavior, and some Intel CPUs are actually
quite close to that performance model. But other CPU behaviors can be observed
in the wild. Consider instead a CPU whose floating-point ALUs have two operating
modes:

- In "normal mode", an ALU processes normal floating-point numbers at optimal
  speed, but it cannot process subnormal numbers. When a subnormal number is
  encountered, the ALU must perform an expensive switch to an alternate generic
  operating mode in order to process it.
- In this alternate "generic mode", the ALU can process both normal and
  subnormal numbers, but operates with reduced efficiency (e.g. it carefully
  checks the exponent of the number before each multiplication, instead of
  speculatively computing the normal result and later discarding it if the
  number turns out to actually be subnormal).
- Because of this reduced efficiency and the presumed rarity of subnormal
  inputs, the ALU will spontaneously switch back to the initial "normal mode"
  after some amount of time has elapsed without any subnormal number showing up
  in the input data stream.

Assuming such a more complex performance model, the relationship between the
average processing time and the share of subnormal data in the input would not
be an affine straight line anymore:

- For small amounts of subnormal numbers in the input data stream, each
  subnormal input will cause an expensive switch to "generic mode", followed by
  a period of time during which mostly-normal numbers are processed at a reduced
  data rate, then a return to "normal mode".
  * In this initial phase of low subnormal input density, we expect a mostly
    linear growth of processing overhead, where the initial
    overhead(%subnormals) curve slope is the cost of each ALU normal → generic →
    normal mode round trip, to which we add the number of normal numbers that
    are processed during the generic mode operating period, multiplied by the
    extra overhead of processing a normal number in generic mode.
- As the share of subnormal numbers in the input data stream increases, the
  CPU's ALUs will see more and more subnormal numbers pass by. Eventually, the
  density of subnormal numbers in the input will become high enough to go below
  the ALUs' internal threshold, and cause ALUs to start delaying some generic →
  normal mode switches.
  * This will result in a reduction of the number of normal → generic → normal
    mode round trips. Therefore, if said round trips are a lot more expensive
    than the cost of processing a few normal numbers in generic mode for the
    duration of the round trip, as initially assumed, the overall processing 
    overhead will start to decrease.
- Finally, beyond a certain share of subnormal inputs, the ALUs will be
  constantly operating in generic mode, and the only observable overhead will be
  that of processing normal numbers in the subnormals-friendly generic mode.
  * It should be noted that the associated overhead that the CPU manufacturer
    tries to avoid may not be a mere linear decrease in throughput, but instead
    something more subtle like an increase in operation latency or ALU energy
    consumption.

Such an overhead curve that grows then shrinks and eventually reaches a plateau
as the share of subnormal inputs increases, was indeed observed on AMD Zen 2 and
3 CPUs when processing certain arithmetic operations like MUL. This suggests
that a mechanism along the line of the one described above is at play on those
CPUs.

As you can see, studying the dependence of subnormal number processing overhead
on the share of subnormal numbers in the input data stream can provide valuable
insight into the underlying hardware implementation of subnormal arithmetic.


## Data type

Besides individual single and double precision numbers, many CPUs can process
data in SIMD batches of one or more width. On microarchitectures with a
sufficiently dark and troubled past, we could expect the subnormals fallback
path to behave differently depending on the precision of the floating-point
number that are being manipulated, or the width of the SIMD batches. Such
differences, if any, will reveal more limitations of the CPU's subnormal
processing path.
