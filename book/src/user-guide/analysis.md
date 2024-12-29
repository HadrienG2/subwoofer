# Analysis

At the time of writing, Subwoofer does not produce fully digested data telling
you e.g. what slowdown you can expect when multiplying subnormal numbers instead
of normal numbers in throughput-bound numerical code that operates from inputs
that reside in the L1 CPU cache.

Instead, this information must be obtained through a manual analysis process. In
the remainder of this documentation, we will explain to you how this process is
performed, then show you examples of the kind of results that you can obtain on
the particular CPUs that the authors got access to.
