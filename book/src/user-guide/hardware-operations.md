# Estimating hardware performance

The _raison d'être_ of Subwoofer is to study how basic hardware floating-point
arithmetic operations behave in presence of a stream of data that contains a
certain share of subnormal numbers, in both latency-bound and throughput-bound
configurations. This is not easy as it seems because...

- To study the performance of latency-bound operations, we need long dependency
  chains made of many copies of the same operation, where each operation takes
  the output of the previous operation as one of its inputs.
- To enforce a share of subnormal inputs othen than 100%, we must ensure that
  the output of operation N, which serves as one of the inputs of operation N+1,
  is a normal number, while other inputs come from a pseudorandom input data
  stream of well-controlled characteristics.
- Many IEEE-754 arithmetic operations produce a non-normal number when fed with
  a normal and subnormal operand. For example, multiplication of a normal number
  by a subnormal number may produce a subnormal output. We need to turn such
  non-normal numbers back into normal numbers before we can use them as the
  input of the next step of the dependency chain, by passing them through an
  auxiliary operation which is...
  * Safe to perform on both normal and subnormal numbers.
  * As cheap as possible to limit the resulting measurement bias.

To achieve these goals, we often need to chain the hardware operation that we
are trying to study with another operation that we also study, chosen to be as
cheap as possible so that the impact on measured performance is minimal. Then we
subtract the impact of that other operation to estimate the impact of the
operation of interest in isolation.

## ADD/SUB

Adding or subtracting a subnormal number to a normal number produces a normal
number, therefore the overhead of ADD/SUB can be studied in isolation. Because
these two operations have identical performance characteristics on all known
hardware, we have a single `add` benchmark that measures the average of ADD
only, and it is assumed that SUB (or addition of the negation) has the same
performance profile.

## MIN/MAX

The maximum of a subnormal and a normal number is a normal number, therefore the
overhead of MAX can be studied in isolation. This is done by the `max`
benchmark. Sadly, MIN does not have this useful property, but its performance
characteristics are the same as those of MAX on all known hardware, so for now
we will just assume that they have the same overhead and that measuring the
performance of MAX is enough to know the performance of MIN.

## MUL

The product of a subnormal and a normal number is a subnormal number, but we can
use a MAX to get back into normal range. This is what the `mul_max` benchmark
does. By subtracting the execution time of `max` from the execution time of
`mul_max`, we get an estimate of the execution time that MUL would have in
isolation, which we can use to estimate the latency and throughput of MUL.

## SQRT

The square root of a subnormal number may or may not be a subnormal number, but
as before we can use a MAX to get back into normal range, and subtract the
overhead of MAX to get an estimate of the overhead of SQRT. Therefore, much like
we had a `mul_max` benchmark, we also have a `sqrt_positive_max` benchmark.

Why "positive", you may ask? Well, for now we only test with positive inputs,
for a few reasons:

- Computing square roots of negative numbers is normally a math error,
  well-behaved programs shouldn't do that in a loop in their hot code path. So
  the performance of the error path isn't that important.
- The square root of a negative number is a NaN, and going back from a NaN to a
  normal number of a reasonable order of magnitude without breaking the
  dependency chain is a lot messier than going from a subnormal to a normal
  number (can't just use a MAX, need to play weird tricks with the exponent bits
  of the underlying IEEE-754 representation, which may cause unexpected CPU
  overhead linked to integer/FP domain crossing).
- The negative argument path of the libm `sqrt()` function that people actually
  use is often partially or totally handled in software, so getting to the
  hardware overhead is difficult, and even if we manage it won't be
  representative of typical real-world performance.

## DIV

Division is interesting because it is one of the few basic IEEE-754 binary
arithmetic operations where the two input operands play a highy asymmetrical
role:

- If we divide possibly subnormal inputs by a normal number, and use the output
  as the denominator of the next division, then we are effectively doing the
  same as multiplying a possibly subnormal number by the inverse of a normal
  number, which is another normal number. As a result, we end up with a pattern
  that is quite similar to that of the `mul_max` benchmark, and again we can use
  MAX as a cheap mechanism to recover from subnormal outputs. This is how the
  `div_numerator_max` benchmark works.
- If we divide a normal number by possibly subnormal inputs, and use the output
  as the numerator of the next division, then the main IEEE-754 special case
  that we need to guard against is not subnormal outputs but infinite outputs.
  This can be done using a MIN that takes infinities back into normal range, and
  that is what the `div_denominator_min` benchmark does. As discussed above, to
  analyze this benchmark, we will assume that MIN has the same performance
  characteristics as MAX and use the results that we collected for MAX.

## FMA

Because Fused Multiply-Add (FMA) has three operands that play two different
roles (multiplier or addend), we have more freedom in how we set up a dependency
chains of FMA with a feedback path from the output of operation N to one of the
inputs of operation N+1. This is what we chose:

* In benchmark `fma_multiplier`, the input data is fed to a multiplier argment
  of the FMA, multiplied by a constant factor, and alternatively added to and
  subtracted from an accumulator. This is effectively the same as the `add`
  benchmark, just with a larger or smaller step size, so for this pattern we can
  study the overhead of FMA with possibly subnormal multipliers in isolation,
  without taking corrective action to guard against non-normal outputs.
* In benchmark `fma_addend`, input data is fed to the addend argument of the
  FMA, and we add to it the current accumulator multiplied by a constant factor.
  The result then becomes the next accumulator. Unfortunately, depending on how
  we pick the constant factor, this factor is doomed to eventually overflow or
  underflow in some input configurations:

  - If the constant factor is >= 1 or sufficiently close to 1, then for a stream
    of normal inputs the value of the accumulator will experience unbounded
    growth and eventually overflow.
  - If the constant factor is < 1, then for a stream of subnormal inputs the
    value of the acccumulator will decay and eventually become subnormal.

  To prevent this, we actually alternate between multiplying by the chosen
  constant factor and its inverse. This should not meaningfully affect that
  measured performance characteristics.
* In benchmark `fma_full_max`, two substreams of the input data are fed to the
  addend argument of the FMA and one of its multiplier argument, with the
  feedback path taking the other multiplier argument. This configuration allows
  us to check if the CPU has a particularly hard time with FMAs that produce
  subnormal outputs. But precisely because we can get subnormal results, we need
  a MAX to bring the accumulator back to normal range.
