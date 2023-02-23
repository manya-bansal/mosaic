#!/bin/bash

#SBATCH -J sddmm_bench
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manya227@stanford.edu
#SBATCH --nodes=1
#SBATCH --mem MaxMemPerNode
#SBATCH --exclusive
#SBATCH --time=10:00:00


systems=("mkl" "taco" "gsl" "blas" "tblis")

SUITSPARSE_INPUT=exdata_1

mkdir $SUITSPARSE_INPUT
for i in "${systems[@]}"
do 
    /home/ubuntu/taco-interoperable/build/bin/./taco-bench --benchmark_filter=bench_suitesparse_sddmm_$i  --benchmark_format=json --benchmark_out=/home/ubuntu/taco-interoperable/bench/suitesparse_sddmm/$SUITSPARSE_INPUT/$i
done