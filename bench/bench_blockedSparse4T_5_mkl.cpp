#include "bench.h"
#include "benchmark/benchmark.h"
#include <random>


#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/mkl_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_interface/gsl_interface.h"
#include "taco/storage/file_io_mtx.h"

using namespace taco;

extern bool gsl_compile;

static std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

static void bench_blockedSparse4T_5_mkl(benchmark::State& state) {

   int dim = state.range(0);
  
   Tensor<float> B("B", {dim, dim, dim, dim}, Format{Sparse, Dense, Sparse, Dense});
   Tensor<float> C("C", {dim, dim, dim, dim}, Format{Dense, Dense, Dense, Dense});

   std::mt19937 mt(0); 
   
   float SPARSITY = .05;
   for (int i = 0; i < dim; i++) {
    for (int k = 0; k < dim; k++) {
      const float randnum = mt();
      float rand_float = randnum/(float)(mt.max());
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
    stmt = stmt.holdConstant(new MklMM(), accelerateExpr, {j, m, i}, precomputed(i, k, m, n));
    

    A.compile(stmt);
    A.assemble();
    auto func = A.compute_split();
    auto pair = A.returnFuncPackedRaw(func);
    state.ResumeTiming();
    pair.first(func.data());
  }

}

TACO_BENCH(bench_blockedSparse4T_5_mkl)->DenseRange(10, 100, 5);

