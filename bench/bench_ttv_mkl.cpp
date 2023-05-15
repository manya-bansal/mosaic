#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_interface/mkl_interface.h"

using namespace taco;

static void bench_ttv_mkl(benchmark::State& state) {
    int dim = state.range(0);
   
  
  
   Tensor<float> B("B", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> C("C", {dim}, Format{Dense});
   TensorVar precomputed("precomputed", Type(taco::Float32, {Dimension(dim), Dimension(dim)}), Format{Dense, Dense});

   IndexVar i("i");
   IndexVar j("j");
    IndexVar k("k");

    IndexExpr accelerateExpr = B(i, j, k) * C(k);


    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        for (int k = 0; k < dim; k++) {
          B.insert({i, j, k}, (float) i + j + k);
        }
      }
   }

   B.pack();
   for (int i = 0; i < dim; i++) {
      C.insert({i}, (float) i);
   }
   
  C.pack();
   
  
  
   for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> A("A", {dim, dim}, {Dense, Dense});
    A(i, j) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.holdConstant(new MklSgemv(), accelerateExpr, {i}, precomputed(i, j));

    A.compile(stmt);
    A.assemble();
    state.ResumeTiming();
    A.compute();
  }
}

TACO_BENCH(bench_ttv_mkl)->DenseRange(10, 100, 10);
