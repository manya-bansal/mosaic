#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_interface/gsl_interface.h"

using namespace taco;

static void bench_mm_taco(benchmark::State& state) {
    int dim = state.range(0);
   
  
   Tensor<float> B("B", {dim, dim}, Format{Dense, Dense});
   Tensor<float> C("C", {dim, dim}, Format{Dense, Dense});
   Tensor<float> D("C", {dim, dim}, Format{Dense, Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = B(i, j) * C(j, k);
  
  
   for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> A("A", {dim, dim}, Format{Dense, Dense});
    A(i, k) = accelerateExpr;

    A.compile();
    A.assemble();
    state.ResumeTiming();
    A.compute();
  }
}

TACO_BENCH(bench_mm_taco)->DenseRange(1000, 5000, 100);

