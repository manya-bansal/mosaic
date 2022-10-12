#!/bin/bash

systems=("taco" "tblis")
# systems=("gsl")

for i in "${systems[@]}"
do
   /home/manya227/taco-interoperable/build/bin/./taco-bench --benchmark_filter=bench_ttm_$i  --benchmark_format=json --benchmark_out=/home/manya227/taco-interoperable/bench/ttm/result/$i
done
