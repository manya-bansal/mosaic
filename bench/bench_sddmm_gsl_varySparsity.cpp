#include "bench.h"
#include "benchmark/benchmark.h"
#include <sstream>

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_interface/gsl_interface.h"
#include "taco/storage/file_io_mtx.h"

#define DIM_SIZE 2000

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


static void bench_sddmm_varySparisty_gsl(benchmark::State& state, float SPARSITY, int dim) {
  gsl_compile = true;

  int NUM_I = dim;
  int NUM_K = dim;
  int NUM_J = dim;
  
  Tensor<float> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<float> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<float> D("D", {NUM_J, NUM_K}, {Dense, Dense});

  std::map<float, std::string> floatToString = { {0.00625, "0.00625"}, {0.0125, "0.0125"}, {0.025, "0.025"},
                                                  {0.05, "0.05"}, {0.1, "0.1"}, {0.2, "0.2"}, {0.4, "0.4"},
                                                  {0.6, "0.6"}, {0.8, "0.8"}, {1.0, "1.0"}};

  std::string generateData = "python3 /home/ubuntu/mosaic/data/data_gen.py --bench sddmm_sp --dim ";
  generateData += std::to_string(dim);
  generateData += " --nnz ";
  generateData += floatToString[SPARSITY];
  generateData += " --out_dir /home/ubuntu/mosaic/data/spdata/";
  exec(generateData.c_str());
  std::string filename = "/home/ubuntu/mosaic/data/spdata/sddmm_sp/B_"+ std::to_string(dim) + "_" + floatToString[SPARSITY] + ".mtx";
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

    IndexStmt stmt = A.getAssignment().concretize();
    stmt = stmt.accelerate(new GSLMM(), accelerateExpr);
    A.compile(stmt);
    A.assemble();
    auto func = A.compute_split();
    auto pair = A.returnFuncPackedRaw(func);
    state.ResumeTiming();
    pair.first(func.data());
  }

  gsl_compile = false;
  std::string eraseData = "rm -rf /home/ubuntu/mosaic/data/spdata/sddmm_sp/B_" + std::to_string(dim) + "_" + floatToString[SPARSITY] + ".mtx";
  exec(eraseData.c_str());
}

TACO_BENCH_ARGS(bench_sddmm_varySparisty_gsl, 0.00625, 0.00625, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gsl, 0.0125,  0.0125, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gsl, 0.025,   0.025, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gsl, 0.05,    0.05, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gsl, 0.1,     0.1, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gsl, 0.2,     0.2, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gsl, 0.4,     0.4, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gsl, 0.6,     0.6, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gsl, 0.8,     0.8, DIM_SIZE);
TACO_BENCH_ARGS(bench_sddmm_varySparisty_gsl, 1,       1, DIM_SIZE);


