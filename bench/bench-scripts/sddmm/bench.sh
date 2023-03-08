#!/bin/bash
systems=("mkl" "gsl" "dot_blas" "taco" "blas" "tblis" "gemv_blas")

for i in "${systems[@]}"
do  
    $PATH_TO_MOSAIC_ARTIFACT/mosaic/build/bin/./taco-bench --benchmark_filter=bench_sddmm_$i  --benchmark_format=json --benchmark_out=/home/reviewer/mosaic/bench/bench-scripts/sddmm/result/$i
done