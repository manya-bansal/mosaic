#!/bin/bash
systems=("blas" "taco" "mkl" "gsl")

for i in "${systems[@]}"
do  
    /home/reviewer/mosaic/build/bin/./taco-bench --benchmark_filter=bench_blockedSparse4T_5_$i  --benchmark_format=json --benchmark_out=/home/reviewer/mosaic/bench/bench-scripts/blockedSparse_4T_5/result/$i
done