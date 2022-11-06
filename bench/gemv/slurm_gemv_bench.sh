#!/bin/bash

#SBATCH -J sddmm_var_sparsity_bench
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manya227@stanford.edu
#SBATCH --nodes=1
#SBATCH --mem MaxMemPerNode
#SBATCH --exclusive
#SBATCH --time=10:00:00

systems=("blas" "taco" "gsl" "mkl" "tblis")

for i in "${systems[@]}"
do  
    /home/ubuntu/taco-interoperable/build/bin/./taco-bench --benchmark_filter=bench_gemv_$i  --benchmark_format=json --benchmark_out=/home/ubuntu/taco-interoperable/bench/gemv/result/$i
done