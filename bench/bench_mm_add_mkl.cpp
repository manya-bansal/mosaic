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

#define DIM_SIZE 200

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

static void bench_mm_add_mkl(benchmark::State& state, float SPARSITY, int dim) {

  int NUM_I = dim;
  int NUM_K = dim;
  int NUM_J = dim;
  
  Tensor<float> B("B", {NUM_I, NUM_K}, {taco::Sparse, taco::Dense});
  Tensor<float> C("C", {NUM_I, NUM_J}, {taco::Sparse, taco::Dense});

  std::ostringstream ss;
  ss << SPARSITY;
  std::string generateData = "python3 /home/ubuntu/mosaic/data/data_gen.py --bench mmAdd --dim ";
  generateData += std::to_string(dim);
  generateData += " --nnz ";
  generateData += ((SPARSITY == 1) ? "1.0" : ss.str());
  generateData += " --out_dir /home/ubuntu/mosaic/data/spdata/";
  exec(generateData.c_str());
  std::string filename = "/home/ubuntu/mosaic/data/spdata/mmAdd/B_"+ std::to_string(dim) + "_" +  ((SPARSITY == 1.0) ? "1.0" : ss.str()) + ".mtx";
  std::cout <<  filename << std::endl;
   std::cout <<  ((SPARSITY == 1.0) ? "1.0" : ss.str())  << std::endl;
  B = castToType<float>("B", readMTX(filename, CSR));
  filename = "/home/ubuntu/mosaic/data/spdata/mmAdd/C_"+ std::to_string(dim) + "_" +  ((SPARSITY == 1.0) ? "1.0" : ss.str()) + ".mtx"; 
  C = castToType<float>("C", readMTX(filename, CSR)); 

 

  IndexVar i("i"), j("j"), k("k");

  IndexExpr accelerateExpr = B(i,j) + C(i, j);

   for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> A("A", {NUM_I, NUM_K}, {Dense, Dense}, 0);
    A(i,j) = accelerateExpr;
    IndexStmt stmt = A.getAssignment().concretize();
    stmt = stmt.accelerate(new MklAdd(), accelerateExpr);

    A.compile();
    A.assemble();
    auto func = A.compute_split();
    auto pair = A.returnFuncPackedRaw(func);
    state.ResumeTiming();
    pair.first(func.data());
  }
  gsl_compile = false;
  std::string eraseData = "rm -rf /home/ubuntu/mosaic/data/spdata/mmAdd/B_" + std::to_string(dim) + "_" +  ((SPARSITY == 1.0) ? "1.0" : ss.str()) + ".mtx";
  exec(eraseData.c_str());
  eraseData = "rm -rf /home/ubuntu/mosaic/data/spdata/mmAdd/C_" + std::to_string(dim) + "_" + ((SPARSITY == 1.0) ? "1.0" : ss.str()) + ".mtx";
  exec(eraseData.c_str());
  gsl_compile = false;
}

TACO_BENCH_ARGS(bench_mm_add_mkl, 0.00625, 0.00625, DIM_SIZE);
TACO_BENCH_ARGS(bench_mm_add_mkl, 0.0125, 0.0125, DIM_SIZE);
TACO_BENCH_ARGS(bench_mm_add_mkl, 0.025, 0.025, DIM_SIZE);
TACO_BENCH_ARGS(bench_mm_add_mkl, 0.05, 0.05, DIM_SIZE);
TACO_BENCH_ARGS(bench_mm_add_mkl, 0.1,  0.1, DIM_SIZE);
TACO_BENCH_ARGS(bench_mm_add_mkl, 0.2,   0.2, DIM_SIZE);
TACO_BENCH_ARGS(bench_mm_add_mkl, 0.4,    0.4, DIM_SIZE);
TACO_BENCH_ARGS(bench_mm_add_mkl, 0.8,     0.8, DIM_SIZE);
TACO_BENCH_ARGS(bench_mm_add_mkl, 1.0,     1.0, DIM_SIZE);

