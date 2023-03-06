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

systems=("taco" "mkl")

for i in "${systems[@]}"
do  
    /home/reviewer/mosaic/build/bin/./taco-bench --benchmark_filter=bench_mm_add_$i  --benchmark_format=json --benchmark_out=/home/reviewer/mosaic/bench/bench-scripts/mmadd/result/$i
done