# Suggestions

Like all CPU microbenchmarks, Subwoofer is affected by various external factors
including your operating system's power management strategy and resource-sharing
interference with other background processes running on the same machine.

We use [Criterion](https://bheisler.github.io/criterion.rs/book/) as a robust
benchmark harness that tries it best to work around these irrelevant effects,
but if you have the time, you can also do your part to improve result quality by
following usual CPU microbenchmarking recommendations.

These recommendations are listed below, in rough order of decreasing importance.

1. Before running any benchmark, you should shut down any background OS process
   that is not required by the benchmark, and keep background processes to a
   minimum until the benchmark is done performing timing measurements.
   * If you do not do this, even intermittent background tasks may keep other
     CPU cores busy, which will lead the CPU core on which the benchmark is
     running to downclock in unpredictable ways under a normal CPU frequency
     scaling policy. See below for more info on how you can turn that off, and
     why you may not want do do so.
   * More CPU-intensive background tasks may steal CPU time from the benchmark,
     which will more affect timing measurements. But because this benchmark is
     single-threaded, you have a fair bit of headroom before this starts to be a
     problem.
   * Background tasks may also put pressure on resources which are shared
     between CPU cores, like the CPU-RAM interconnect, which will affect a few
     specific benchmarks that are not enabled by default.
2. If the computer you are testing is a laptop, it should be plugged into an
   electrical outlet.
   * The CPU performance of some laptops has been observed to fluctuate in
     highly unpredictable ways when operating on battery power. The exact cause
     for this phenomenon is not fully understood, but it may be related to
     maximal current draw limitations of the underlying laptop battery.
3. OS and vendor performance/powersaving tunables should be set up for maximal
   performance.
   * If you do not do this, your benchmark results will exhibit some dependence
     on the powersaving algorithm used by your computer. This is not ideal
     because the details of this powersaving algorithm may depend on many
     things: installed hardware and operating system, version of CPU microcode
     and all software involved in CPU power management decision...
4. If maximal output reproducibility is desired, you can also disable "turbo"
   frequency scaling and force your CPU to constantly operate at its nominal
   frequency.
   * On Linux, this used to be easily done with vendor-agnostic tools like
     [`cpupower`](https://linux.die.net/man/1/cpupower), but modern AMD and
     Intel CPUs have thrown a wrench into this and vendor-specific tools are now
     needed. For Intel CPUs, I recommend
     [`pstate-frequency`](https://github.com/pyamsoft/pstate-frequency). For
     other CPUs, see [this page of the Arch
     wiki](https://wiki.archlinux.org/title/CPU_frequency_scaling#Scaling_drivers).
   * Note that setting up your CPU like this is a nontrivial tradeoff that you
     should consider carefully:
     - As a major benefit, it makes your benchmark output more accurate and
       reproducible.
     - As a minor benefit, it lets you convert criterion's time-based
       measurements into cycle-based measurements, which is appropriate for
       benchmarks where CPU code execution is the limiting factor. Note that
       this is not true of all microbenchmarks provided by Subwoofer, a few of
       them are limited by other resources like the CPU-RAM interconnect, whose
       clock is not in sync with the CPU clock.
     - In exchange for these benefits, the main drawback of this approach is
       that you get performance results that are less representative of
       real-world CPU usage, since computers do not normally run with CPU
       frequency scaling disabled.
