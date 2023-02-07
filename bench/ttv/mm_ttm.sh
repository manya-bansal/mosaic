#!/bin/bash

systems=("tblis")
# systems=("gsl")

for i in "${systems[@]}"
do
   /home/ubuntu/taco-interoperable/build/bin/./taco-bench --benchmark_filter=bench_ttv_$i  --benchmark_format=json --benchmark_out=/home/ubuntu/taco-interoperable/bench/ttv/result/$i
done
