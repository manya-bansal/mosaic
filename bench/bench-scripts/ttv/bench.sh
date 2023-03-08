#!/bin/bash
systems=("taco" "tblis" "gsl" "blas" "mkl")

for i in "${systems[@]}"
do  
    $PATH_TO_MOSAIC_ARTIFACT/mosaic/build/bin/./taco-bench --benchmark_filter=bench_ttv_$i  --benchmark_format=json --benchmark_out=/home/reviewer/mosaic/bench/bench-scripts/ttv/result/$i
done