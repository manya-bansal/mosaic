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

#define SUITSPARSE_INPUT "progas"
#define SUITSPARSE_DATA_PATH "/home/reviewer/mosaic-benchmarks/data/suitesparse"

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

static void bench_suitesparse_sddmm_taco(benchmark::State& state) {
  int dim = state.range(0);
  int NUM_I = dim;
  int NUM_K = dim;
  int NUM_J = dim;
  
  Tensor<float> B;
  std::string filename = SUITSPARSE_DATA_PATH + std::string("/")+ SUITSPARSE_INPUT  + ".mtx";
  B = castToType<float>("B", readMTX(filename, CSR));


  NUM_I = B.getDimension(0);
  // std::cout << "NUM_I " << NUM_I << std::endl;
  NUM_K = B.getDimension(1);
  // std::cout << "NUM_K " << NUM_K << std::endl;
  NUM_J = B.getDimension(0);
  // std::cout << "NUM_J " << NUM_J << std::endl;
  dim = NUM_I;

  Tensor<float> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<float> D("D", {NUM_J, NUM_K}, {Dense, Dense});


  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      C.insert({i, j}, (float) 2);
    }
  }


  for (int i = 0; i < NUM_J; i++) {
    for (int j = 0; j < NUM_K; j++) {
      D.insert({i, j}, (float) 2);
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

    IndexStmt stmt = A.getAssignment().concretize();

    A.compile(stmt);
    A.assemble();
    auto func = A.compute_split();
    auto pair = A.returnFuncPackedRaw(func);
    state.ResumeTiming();
    pair.first(func.data());
  }
}


TACO_BENCH(bench_suitesparse_sddmm_taco)->DenseRange(100, 200, 200);

