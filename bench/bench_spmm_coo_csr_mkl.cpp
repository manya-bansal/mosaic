#include "bench.h"
#include "benchmark/benchmark.h"
#include <sstream>

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_interface/gsl_interface.h"
#include "taco/accelerator_interface/mkl_interface.h"
#include "taco/storage/file_io_mtx.h"

#define DIM_SIZE 4000

using namespace taco;

static void bench_coo_csr_spmm_mkl(benchmark::State& state, float SPARSITY, int dim) {

  int NUM_I = dim;
  int NUM_K = dim;
  int NUM_J = dim;
  
  Tensor<float> B("B", {NUM_I, NUM_K}, COO(2));
  Tensor<float> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  
  std::mt19937 mt(0); 

   for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      const float randnum = mt();
      float rand_float = randnum/(float)(mt.max());
         if (rand_float < SPARSITY) {
                     B.insert({i, k}, (float) (float) 100);
      }
    }
  }

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      C.insert({i, j}, (float) i+j);
    }
  }

  B.pack();
  C.pack();

  IndexVar i("i"), j("j"), k("k");
  IndexExpr accelerateExpr = B(i,j) * C(j,k);

  for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> A("A", {NUM_I, NUM_K}, {Dense, Dense}, 0);
    A(i,k) = accelerateExpr;

    IndexStmt stmt = A.getAssignment().concretize();
    stmt = stmt.accelerate(new SparseMklMMCOOCSR(), accelerateExpr);

    A.compile(stmt);
    A.assemble();
    auto func = A.compute_split();
    auto pair = A.returnFuncPackedRaw(func);
    state.ResumeTiming();
    pair.first(func.data());
  }
}


TACO_BENCH_ARGS(bench_coo_csr_spmm_mkl, 0.00078125, 0.00078125, DIM_SIZE);
TACO_BENCH_ARGS(bench_coo_csr_spmm_mkl, 0.0015625, 0.0015625, DIM_SIZE);
TACO_BENCH_ARGS(bench_coo_csr_spmm_mkl, 0.003125, 0.003125, DIM_SIZE);
TACO_BENCH_ARGS(bench_coo_csr_spmm_mkl, 0.00625, 0.00625, DIM_SIZE);
TACO_BENCH_ARGS(bench_coo_csr_spmm_mkl, 0.0125,  0.0125, DIM_SIZE);
TACO_BENCH_ARGS(bench_coo_csr_spmm_mkl, 0.025,   0.025, DIM_SIZE);
TACO_BENCH_ARGS(bench_coo_csr_spmm_mkl, 0.05,    0.05, DIM_SIZE);
TACO_BENCH_ARGS(bench_coo_csr_spmm_mkl, 0.1,     0.1, DIM_SIZE);
TACO_BENCH_ARGS(bench_coo_csr_spmm_mkl, 0.2,     0.1, DIM_SIZE);
TACO_BENCH_ARGS(bench_coo_csr_spmm_mkl, 0.4,     0.1, DIM_SIZE);
TACO_BENCH_ARGS(bench_coo_csr_spmm_mkl, 0.8,     0.1, DIM_SIZE);

