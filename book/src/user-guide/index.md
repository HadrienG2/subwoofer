# User guide

Welcome to the documentation of
[Subwoofer](https://github.com/hadrienG2/subwoofer)! This project provides a set
of microbenchmarks that lets you review how [subnormal
numbers](https://en.wikipedia.org/wiki/Subnormal_number) affect the performance
of floating-point arithmetic operations on your CPU's microarchitecture.

Currently supported arithmetic includes ADD, SUB, MUL, FMA, and SQRT of positive
numbers, mainly with subnormal inputs and sometimes with subnormal outputs too.
DIV is not supported yet, mainly because I haven't figured out a minimally
invasive way to automatically reset an accumulator after division by a subnormal
number (which saturates the accumulator to +inf). Suggestions welcome!

As the time of writing, this benchmark has only been rigorously checked for
correctness on x86_64. But it has been designed with due consideration for other
common CPU microarchitectures, so I believe that given a week or two of
interactive access to an ARM or RISC-V machine with perf profiler support, I
should be able to validate/debug it for those ISAs too.
