#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"

using namespace taco;

static void bench_sdmm_taco(benchmark::State& state) {
  int dim = state.range(0);
   
  int NUM_I = dim;
  int NUM_K = dim;
  int NUM_J = dim;

  float SPARSITY = .3;
  
  Tensor<double> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<double> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<double> D("D", {NUM_J, NUM_K}, {Dense, Dense});

  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  IndexVar i("i"), j("j"), k("k");

  IndexExpr accelerateExpr = B(i,k) * C(i,j) * D(j,k);

  for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<double> A("A", {NUM_I, NUM_K}, {Dense, Dense});
    A(i,k) = accelerateExpr;
    A.compile();
    A.assemble();
    state.ResumeTiming();
    A.compute();
  }
}

TACO_BENCH(bench_sdmm_taco)->DenseRange(100, 10000, 200);

