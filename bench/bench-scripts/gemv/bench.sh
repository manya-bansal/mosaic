

systems=("blas" "gsl" "tblis" "taco" "mkl")

for i in "${systems[@]}"
do  
    /home/reviewer/mosaic/build/bin/./taco-bench --benchmark_filter=bench_gemv_$i  --benchmark_format=json --benchmark_out=/home/reviewer/mosaic/bench/bench-scripts/gemv/result/$i
done