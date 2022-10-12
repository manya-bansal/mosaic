#!/bin/bash

#SBATCH -J saxpy_bench
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manya227@stanford.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00

systems=("blas")

srun /home/manya227/taco-interoperable/build/bin/./taco-bench --benchmark_filter=bench_saxpy_${systems[0]}  --benchmark_format=json --benchmark_out=/home/manya227/taco-interoperable/bench/saxpy/result/${systems[0]}