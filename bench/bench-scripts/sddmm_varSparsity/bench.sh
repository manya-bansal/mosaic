#!/bin/bash
systems=("dot_mkl" "mkl" "taco" "blas" "gemv_mkl" "tblis" "gemv_gsl")

for i in "${systems[@]}"
do  
    $PATH_TO_MOSAIC_ARTIFACT/mosaic/build/bin/./taco-bench --benchmark_filter=bench_sddmm_varySparisty_$i  --benchmark_format=json --benchmark_out=/home/reviewer/mosaic/bench/bench-scripts/sddmm_varSparsity/result/$i
done