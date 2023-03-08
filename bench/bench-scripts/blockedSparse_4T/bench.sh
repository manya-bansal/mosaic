#!/bin/bash
systems=("blas" "taco" "mkl" "gsl")

for i in "${systems[@]}"
do  
    $PATH_TO_MOSAIC_ARTIFACT/mosaic/build/bin/./taco-bench --benchmark_filter=bench_blockedSparse4T_$i  --benchmark_format=json --benchmark_out=/home/reviewer/mosaic/bench/bench-scripts/blockedSparse_4T/result/$i
done