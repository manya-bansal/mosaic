#!/bin/bash

systems=("gsl")

for i in "${systems[@]}"
do
   /home/manya227/taco-interoperable/build/bin/./taco-bench --benchmark_filter=bench_mm_$i  --benchmark_format=json --benchmark_out=/home/manya227/taco-interoperable/bench/mm/result/$i
done
