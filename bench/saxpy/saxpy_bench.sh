#!/bin/bash

systems=("blas" "gsl" "taco" "tblis")

for i in "${systems[@]}"
do
   /home/manya227/taco-interoperable/build/bin/./taco-bench --benchmark_filter=bench_saxpy_$i  --benchmark_format=json --benchmark_out=/home/manya227/taco-interoperable/bench/saxpy/result/$i
done
