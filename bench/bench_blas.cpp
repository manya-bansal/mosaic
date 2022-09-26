#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/cblas_saxpy.h"

using namespace taco;

static void bench_cblas_saxpy(benchmark::State& state) {
  int dim = state.range(0);

  std::cout << "dim " << dim << std::endl;
   
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
    IndexStmt stmt = A.getAssignment().concretize();
    stmt = stmt.accelerate(new Saxpy(), accelerateExpr);
    A.compile(stmt);
    A.assemble();
    state.ResumeTiming();
    A.compute();
  }
}

TACO_BENCH(bench_cblas_saxpy)->DenseRange(20, 1024, 128);

