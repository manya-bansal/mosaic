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

#define DIM_SIZE 2000

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


static void bench_sddmm_varySparisty_dot_mkl(benchmark::State& state, float SPARSITY, int dim) {
  int NUM_I = dim;
  int NUM_K = dim;
  int NUM_J = dim;
  
  Tensor<float> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<float> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<float> D("D", {NUM_J, NUM_K}, {Dense, Dense});



  std::string generateData = "python3 /home/reviewer/mosaic-benchmarks/data/data_gen.py --bench sddmm_sp --dim ";
  generateData += std::to_string(dim);
  generateData += " --nnz ";
  std::ostringstream ss;
  ss << SPARSITY;generateData += ss.str();
  generateData += " --out_dir /home/reviewer/mosaic-benchmarks/data/spdata/";
  exec(generateData.c_str());
  std::string filename = "/home/reviewer/mosaic-benchmarks/data/spdata/sddmm_sp/B_"+ std::to_string(dim) + "_" + ss.str() + ".mtx";
  B = castToType<float>("B", readMTX(filename, CSR));

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      C.insert({i, j}, (float) i+j);
      D.insert({i, j}, (float) i+j);
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
    std::cout << "here" << std::endl;
    stmt = stmt.holdConstant(new MklDot(), accelerateExpr, {i, k}, precomputed(i, k));

    A.compile(stmt);
    A.assemble();
    
    auto func = A.compute_split();
    auto pair = A.returnFuncPackedRaw(func);
    std::cout << "start" << std::endl;
    state.ResumeTiming();
    pair.first(func.data());
  }
  std::string eraseData = "rm -rf /home/reviewer/mosaic-benchmarks/data/spdata/sddmm_sp/B_" + std::to_string(dim) + "_" + ss.str() + ".mtx";
  exec(eraseData.c_str());
}

TACO_BENCH_ARGS(bench_sddmm_varySparisty_dot_mkl, 0.00078125, 0.00078125, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_dot_mkl, 0.0015625,  0.0015625, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_dot_mkl, 0.003125,   0.003125, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_dot_mkl, 0.00625,    0.00625, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_dot_mkl, 0.0125,     0.0125, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_dot_mkl, 0.025,     0.025, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_dot_mkl, 0.05,     0.05, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_dot_mkl, 0.1,     0.1, DIM_SIZE);


