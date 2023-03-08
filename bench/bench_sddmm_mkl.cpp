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

static void bench_sddmm_mkl(benchmark::State& state) {
  int dim = state.range(0);
  int NUM_I = dim;
  int NUM_K = dim;
  int NUM_J = dim;
  
  Tensor<float> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<float> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<float> D("D", {NUM_J, NUM_K}, {Dense, Dense});

  char const *ret = getenv("PATH_TO_MOSAIC_ARTIFACT");
  if (!ret) {
    taco_uerror << "Please set the environment variable PATH_TO_MOSAIC_ARTIFACT."
    << "To do so, run (in the mosaic-artifact/scripts dir): source mosaic_env_var.sh.";
  }

 std::string path_to_artifact = std::string(ret);

  std::string generateData = "python3  " + path_to_artifact + "/mosaic-benchmarks/data/data_gen.py --bench sddmm_dim --dim ";
  generateData += std::to_string(dim);
  generateData += " --nnz 0.4";
  generateData += " --out_dir " + path_to_artifact + "/mosaic-benchmarks/data/spdata/";
  exec(generateData.c_str());
  std::string filename = "" + path_to_artifact + "/mosaic-benchmarks/data/spdata/sddmm_dim/B_"+ std::to_string(dim) + "_0.4"  + ".mtx";
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
    stmt = stmt.accelerate(new MklMM(), accelerateExpr);

    A.compile(stmt);
    A.assemble();
    auto func = A.compute_split();
    auto pair = A.returnFuncPackedRaw(func);
    state.ResumeTiming();
    pair.first(func.data());
  }
  std::string eraseData = "rm -rf " + path_to_artifact + "/mosaic-benchmarks/data/spdata/sddmm_dim/B_" + std::to_string(dim) + "_0.4"+ ".mtx";
  exec(eraseData.c_str());
}


TACO_BENCH(bench_sddmm_mkl)->DenseRange(100, 2000, 200);

