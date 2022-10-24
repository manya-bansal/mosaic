#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_interface/gsl_interface.h"

#define DIM_SIZE 1000

using namespace taco;

extern bool gsl_compile;

static void bench_sddmm_varySparisty_gemv_gsl(benchmark::State& state, float SPARSITY, int dim) {
  gsl_compile = true;

  int NUM_I = dim;
  int NUM_K = dim;
  int NUM_J = dim;
  
  Tensor<float> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<float> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<float> D("D", {NUM_J, NUM_K}, {Dense, Dense});

  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (float) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  IndexVar i("i"), j("j"), k("k");

  IndexExpr accelerateExpr = C(i,j) * D(j,k);

   for (auto _ : state) {
    // Setup.
     state.PauseTiming();
    Tensor<float> A("A", {NUM_I, NUM_K}, {Dense, Dense}, 0);
    A(i,k) =  B(i,k) * accelerateExpr;
    TensorVar precomputed("precomputed", Type(taco::Float32, {(size_t)NUM_I, (size_t)NUM_K}), Format{Dense, Dense});

    IndexStmt stmt = A.getAssignment().concretize();
    stmt = stmt.holdConstant(new GSLGemv(), accelerateExpr, {k}, precomputed(i, k));

    A.compile(stmt);
    A.assemble();
    state.ResumeTiming();
    A.compute();
  }

  gsl_compile = false;
}

TACO_BENCH_ARGS(bench_sddmm_varySparisty_gemv_gsl, 0.00625, 0.00625, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gemv_gsl, 0.0125,  0.0125, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gemv_gsl, 0.025,   0.025, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gemv_gsl, 0.05,    0.05, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gemv_gsl, 0.05,    0.05, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gemv_gsl, 0.1,     0.1, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gemv_gsl, 0.2,     0.2, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gemv_gsl, 0.4,     0.4, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gemv_gsl, 0.6,     0.6, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gemv_gsl, 0.8,     0.8, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gemv_gsl, 1,       1, DIM_SIZE);


