#!/bin/bash
systems=("5_blas" "5_taco" "5_mkl" "5_gsl")

for i in "${systems[@]}"
do  
    /home/reviewer/mosaic/build/bin/./taco-bench --benchmark_filter=bench_blockedSparse4T_$i  --benchmark_format=json --benchmark_out=/home/reviewer/mosaic/bench/bench-scripts/blockedSparse_4T_5/result/$i
done