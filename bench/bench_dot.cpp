#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"

using namespace taco;

static void bench_dot(benchmark::State& state) {
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

   IndexExpr accelerateExpr = B(j) * C(j);
  

   for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> A("A", {dim}, Format{Dense});
    IndexVar i("i");
    IndexVar j("j");

    A(i) = sum(j, accelerateExpr);

    A.registerAccelerator(new TblisDot());
   // enable targeting
    A.accelerateOn();

    A.compile();
    A.assemble();
    state.ResumeTiming();
    A.compute();
  }
}

TACO_BENCH(bench_dot)->DenseRange(20, 1000, 20);

