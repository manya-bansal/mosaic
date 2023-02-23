#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/mkl_interface.h"
#include "taco/storage/file_io_mtx.h"

using namespace taco;

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

static void bench_spmv_mkl_col(benchmark::State& state) {
    int dim = state.range(0);
   
    Tensor<float> B("B", {dim, dim}, CSR);
    Tensor<float> C("C", {dim}, Format{Dense});

    float SPARSITY = 0.2;
    int block_size = 10;
    float sparsity_new = SPARSITY;

    for (int i = 0; i < dim; i++) {
        float rand_float = (float)rand()/(float)(RAND_MAX);
        if (rand_float < sparsity_new) {
          for (int j = 0; j < dim; j++ ){
               B.insert({j, i}, (float) ((int) (rand_float*3/sparsity_new)));
          }
      }
    }

    for (int i = 0; i < dim; i++) {
      C.insert({i}, (float)i);
    }

    B.pack();
    C.pack();

    IndexVar i("i");
    IndexVar j("j");
    IndexVar k("k");

    IndexExpr accelerateExpr = B(i, j) * C(j);
  
   for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> A("A", {dim}, Format{Dense});
    A(i) = B(i, j) * C(j);
    IndexStmt stmt = A.getAssignment().concretize();
    stmt = stmt.accelerate(new SparseMklSgemv(), accelerateExpr, true);
    A.compile(stmt);
    A.assemble();
    auto func = A.compute_split();
    auto pair = A.returnFuncPackedRaw(func);
    state.ResumeTiming();
    pair.first(func.data());
  }

}

TACO_BENCH(bench_spmv_mkl_col)->DenseRange(10000, 20000, 1000);

