# Measurements on various hardware

The first column indicates each operation's latency in nanoseconds, and the
throughput in operations/second, when operating on normal inputs. The next
columns indicate the relative slowdown for a given share of subnormal
numbers in the input.

Table cells with a dash cannot be measured with the current setup.


## Intel

## [i9-10900](https://www.intel.fr/content/www/fr/fr/products/sku/199328/intel-core-i910900-processor-20m-cache-up-to-5-20-ghz/specifications.html) (Comet Lake)

### AVX

|            f32x8 |      0% |  25% |  50% |  75% | 100% |
|-----------------:|--------:|-----:|-----:|-----:|-----:|
|          ADD/SUB | 0.93 ns |   1x |   1x |   1x |   1x |
|                  | 8.5 G/s |   1x |   1x |   1x |   1x |
|              MUL | 0.94 ns |   9x |  17x |  25x |  33x |
|                  | 8.5 G/s |  70x | 140x | 210x | 280x |
| FMA (multiplier) | 0.94 ns |   9x |  17x |  25x |  33x |
|                  | 8.5 G/s |  70x | 140x | 209x | 279x |
|     FMA (addend) | 0.93 ns |   9x |  17x |  25x |  33x |
|                  | 8.5 G/s |  71x | 140x | 210x | 280x |
|        FMA (any) | 0.93 ns |  --- |  17x |  --- |  33x |
|                  | 8.5 G/s |  --- | 138x |  --- | 279x |
|             SQRT | ---     |  --- |  --- |  --- |  --- |
|                  | 764 M/s |   8x |  15x |  21x |  28x |

Overall, the impact of subnormals grows linearly as the number of subnormals
increases. It looks like processing any operation other than ADD or SUB on
denormals takes around around 30ns, always, no matter what the operation is or
how many subnormals are present inside of the input. As a result, the relative
overhead will be more or less severe depending on how fast the affected
operation originally was.
