#!/bin/bash

# systems=("block" "row" "col" "random" "block_diagonal")
systems=("row")

for i in "${systems[@]}"
do
   /home/ubuntu/taco-interoperable/build/bin/./taco-bench --benchmark_filter=bench_spmv_mkl_$i  --benchmark_format=json --benchmark_out=/home/ubuntu/taco-interoperable/bench/sparsity_patterns/result/$i
done
