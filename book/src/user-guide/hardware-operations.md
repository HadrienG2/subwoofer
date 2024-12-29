# Estimating hardware performance

The _raison d'être_ of Subwoofer is to study how basic hardware floating-point
arithmetic operations behave in presence of a stream of data that contains a
certain share of subnormal numbers, in both latency-bound and throughput-bound
configurations. This is not easy as it seems because...

- To study the performance of latency-bound operations, we need long dependency
  chains made of many copies of the same operation, where each operation takes
  the output of the previous operation as one of its inputs.
- To enforce a share of subnormal inputs othen than 100%, we must ensure that
  the output of operation N-1, which serves as one of the inputs of operation N,
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

TODO
