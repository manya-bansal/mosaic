#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"

using namespace taco;

static void bench_ttm_taco(benchmark::State& state) {
    int dim = state.range(0); 
   
   Tensor<float> B("B", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> C("C", {dim, dim}, Format{Dense, Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
   IndexVar l("l");

    IndexExpr accelerateExpr = B(i, j, l) * C(k, l);
   
  
  
   for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> A("A", {dim, dim, dim}, Format{Dense, Dense, Dense});
    A(i, j, k) = accelerateExpr;

    A.compile();
    A.assemble();
    state.ResumeTiming();
    A.compute();
  }
}

TACO_BENCH(bench_ttm_taco)->DenseRange(20, 500, 20);

