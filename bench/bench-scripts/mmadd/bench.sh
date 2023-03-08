#!/bin/bash
systems=("taco" "mkl" "bug_mkl")

for i in "${systems[@]}"
do  
    $PATH_TO_MOSAIC_ARTIFACT/mosaic/build/bin/./taco-bench --benchmark_filter=bench_mm_add_$i  --benchmark_format=json --benchmark_out=/home/reviewer/mosaic/bench/bench-scripts/mmadd/result/$i
done