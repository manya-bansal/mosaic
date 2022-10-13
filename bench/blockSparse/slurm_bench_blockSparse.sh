#!/bin/bash

#SBATCH -J saxpy_bench
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manya227@stanford.edu
#SBATCH --nodes=1
#SBATCH --mem=2gb
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00

systems=("tblis")

for i in "${systems[@]}"
do 
    srun /home/manya227/taco-interoperable/build/bin/./taco-bench --benchmark_filter=bench_blockSparse_$i  --benchmark_format=json --benchmark_out=/home/manya227/taco-interoperable/bench/blockSparse/result/$i
done