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

static void bench_blockedSparse4T_taco(benchmark::State& state) {

   int dim = state.range(0);
  
   Tensor<float> B("B", {dim, dim, dim, dim}, Format{Sparse, Dense, Sparse, Dense});
   Tensor<float> C("C", {dim, dim, dim, dim}, Format{Sparse, Dense, Sparse, Dense});

   
   float SPARSITY = .4;
   for (int i = 0; i < dim; i++) {
    for (int k = 0; k < dim; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
         if (rand_float < SPARSITY) {
             for (int m = 0; m < dim; m++) {
                 for (int n = 0; n < dim; n++) {
                     B.insert({i, m, k, n}, (float) (float) 100);
                     C.insert({i, m, k, n}, (float) (float) 100);
                 }
            }
         }
    }
  }

   C.pack();
   B.pack();

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
   IndexVar l("l");
   IndexVar m("m");
   IndexVar n("n");

   IndexExpr accelerateExpr = B(i, k, j, l) * C(j, l, m, n);
   
   for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> A("A", {dim, dim, dim, dim}, Format{Dense, Dense, Dense, Dense});
    TensorVar precomputed("precomputed", Type(taco::Float32, {Dimension(dim), Dimension(dim), Dimension(dim), Dimension(dim)}), Format{Dense, Dense, Dense, Dense});
    A(i, k, m, n) = accelerateExpr;
    IndexStmt stmt = A.getAssignment().concretize();
    A.compile(stmt);
    A.assemble();
    state.ResumeTiming();
    A.compute();
  }

}

TACO_BENCH(bench_blockedSparse4T_taco)->DenseRange(20, 100, 5);

