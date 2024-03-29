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

static void bench_symmgemv_gsl(benchmark::State& state) {

   gsl_compile = true;
     // actual computation
   int dim = state.range(0);
   Tensor<float> b("b", {dim}, Format{Dense});
   Tensor<float> A("A", {dim, dim}, Format{Dense, Dense});

   for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
         A.insert({i, j}, (float) i + j);
      }
   }

   A.pack();
   for (int i = 0; i < dim; i++) {
      b.insert({i}, (float) i);
   }

   IndexVar i("i");
   IndexVar j("j");
   IndexExpr accelerateExpr = A(i, j) * b(j);

   

   for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> res("res", {dim}, Format{Dense});
    res(i) = accelerateExpr;

    IndexStmt stmt = res.getAssignment().concretize();
    stmt = stmt.accelerate(new GSLSymmetricGemv(), accelerateExpr, true);

    res.compile(stmt);
    res.assemble();
    auto func = res.compute_split();
    auto pair = res.returnFuncPackedRaw(func);
    state.ResumeTiming();
    pair.first(func.data());
  }
  gsl_compile = false;

}

TACO_BENCH(bench_symmgemv_gsl)->DenseRange(250, 5000, 250);

