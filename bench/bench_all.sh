#!/bin/bash

./gemv/slurm_gemv_bench.sh
./symmmgemv/slurm_gemv_bench.sh
./spmv/slurm_gemv_bench.sh
./sddmm/slurm_sddmm_bench.sh
./sddmm_varSparsity/slurm_sddmm_var_sparse_bench.sh
