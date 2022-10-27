#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_interface/gsl_interface.h"
#include "taco/accelerator_interface/tensor_interface.h"

using namespace taco;

extern bool gsl_compile;

static void bench_plus3_gsl_tensor(benchmark::State& state) {
    int dim = state.range(0);
   
    gsl_compile = true;

   Tensor<float> B("B", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> C("C", {dim, dim, dim}, Format{Dense, Dense, Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = B(i, j, k) + C(i, j, k);
  
   for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> A("A", {dim, dim, dim}, Format{Dense, Dense, Dense});
    A(i, j, k) = accelerateExpr;
    IndexStmt stmt = A.getAssignment().concretize();
    stmt = stmt.accelerate(new GslTensorPlus(), accelerateExpr, true);
    A.compile(stmt);
    A.assemble();
    state.ResumeTiming();
    A.compute();
  }

  gsl_compile = false;
}

TACO_BENCH(bench_plus3_gsl_tensor)->DenseRange(100, 2000, 100);

