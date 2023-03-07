#!/bin/bash
systems=("stardust")

for i in "${systems[@]}"
do  
    /home/reviewer/mosaic/build/bin/./taco-bench --benchmark_filter=bench_spmv_$i  --benchmark_format=json --benchmark_out=/home/reviewer/mosaic/bench/bench-scripts/spmv/result/$i
done