#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"

using namespace taco;

static void bench_saxpy_tblis(benchmark::State& state) {
  int dim = state.range(0);
   
   Tensor<float> B("B", {dim}, Format{Dense});
   Tensor<float> C("C", {dim}, Format{Dense});
   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   for (int i = 0; i < dim; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

  IndexExpr accelerateExpr = B(i) + C(i);

  for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> A("A", {dim}, Format{Dense});
    IndexVar i("i");
    IndexVar j("j");
    A(i) = accelerateExpr;
    // IndexStmt stmt = A.getAssignment().concretize();
    // stmt = stmt.accelerate(new TblisSaxpy(), accelerateExpr);

    A.registerAccelerator(new TblisSaxpy());
    A.accelerateOn();

    A.compile();
    A.assemble();
    state.ResumeTiming();
    A.compute();
  }
}

TACO_BENCH(bench_saxpy_tblis)->DenseRange(1000, 10000, 200);

