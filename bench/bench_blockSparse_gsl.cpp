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

static void bench_blockSparse_gsl(benchmark::State& state) {

   gsl_compile = true;

   int dim = state.range(0);

  
   Tensor<float> B("B", {dim, dim, dim}, Format{Sparse, Dense, Dense}); 
   Tensor<float> C("C", {dim, dim}, Format{Dense, Dense});

   float SPARSITY = .3;

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }


  for (int i = 0; i < dim; i++) {
   for (int j = 0; j < dim; j++) {
    for (int k = 0; k < dim; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, j, k}, (float) ((int) (rand_float*3/SPARSITY)));
      }
    }
   }
  }


   B.pack();
   C.pack();

  
   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
   IndexVar l("l");
   IndexVar m("m");

   IndexExpr accelerateExpr = B(i, j, l) * C(l, k);
   
   for (auto _ : state) {
    // Setup.
    state.PauseTiming();

    TensorVar precomputed("precomputed", Type(taco::Float32, {Dimension(dim), Dimension(dim), Dimension(dim)}), Format{Dense, Dense, Dense});
    Tensor<float> A("A", {dim, dim, dim}, Format{Dense, Dense, Dense});
    
    A(i, j, k) = accelerateExpr;

    IndexStmt stmt = A.getAssignment().concretize();
    stmt = stmt.holdConstant(new GSLMM(), accelerateExpr, {i}, precomputed(i, j, k));

    A.compile(stmt);
    A.assemble();
    state.ResumeTiming();
    A.compute();
  }

  gsl_compile = false;
}

TACO_BENCH(bench_blockSparse_gsl)->DenseRange(20, 1000, 20);

