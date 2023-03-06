

systems=("blas" "taco" "gsl" "mkl" "tblis")

for i in "${systems[@]}"
do  
    /home/reviewer/mosaic/build/bin/./taco-bench --benchmark_filter=bench_symmgemv_$i  --benchmark_format=json --benchmark_out=/home/reviewer/mosaic/bench/bench-scripts/symmgemv/result/$i
done