#!/bin/bash
systems=("blas" "taco" "gsl" "mkl" "tblis")

for i in "${systems[@]}"
do  
    $PATH_TO_MOSAIC_ARTIFACT/mosaic/build/bin/./taco-bench --benchmark_filter=bench_symmgemv_$i  --benchmark_format=json --benchmark_out=/home/reviewer/mosaic/bench/bench-scripts/symmgemv/result/$i
done