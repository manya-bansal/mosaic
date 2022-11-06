#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_interface/gsl_interface.h"

using namespace taco;

extern bool gsl_compile;

static void bench_sgemm_gsl(benchmark::State& state) {

  gsl_compile = true;
  
  int dim = state.range(0);
   
  
   Tensor<float> B("B", {dim, dim}, Format{Dense, Dense});
   Tensor<float> C("C", {dim, dim}, Format{Dense, Dense});
   Tensor<float> D("C", {dim, dim}, Format{Dense, Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = B(i, j) * C(j, k) ;
  
  
   for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> A("A", {dim, dim}, Format{Dense, Dense});
    A(i, k) = accelerateExpr;
    IndexStmt stmt = A.getAssignment().concretize();
    stmt = stmt.accelerate(new GSLMM(), accelerateExpr, true);
    A.compile(stmt);
    A.assemble();
    auto func = A.compute_split();
    auto pair = A.returnFuncPackedRaw(func);
    state.ResumeTiming();
    pair.first(func.data());
  }

  gsl_compile = false;
}

TACO_BENCH(bench_sgemm_gsl)->DenseRange(100, 1000, 100);

