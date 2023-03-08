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

static void bench_spmv_taco(benchmark::State& state) {
    int dim = state.range(0);
   
    Tensor<float> B("B", {dim, dim}, CSR);
    Tensor<float> C("C", {dim}, Format{Dense});

    char const *ret = getenv("PATH_TO_MOSAIC_ARTIFACT");
  if (!ret) {
    taco_uerror << "Please set the environment variable PATH_TO_MOSAIC_ARTIFACT."
    << "To do so, run (in the mosaic/bench/bench-scripts/ dir): source mosaic_env_var.sh.";
  }

 std::string path_to_artifact = std::string(ret);

  std::string generateData = "python3  " + path_to_artifact + "/mosaic-benchmarks/data/data_gen.py --bench spmv --dim ";
    generateData += std::to_string(dim);
    generateData += " --nnz 0.2 --out_dir " + path_to_artifact + "/mosaic-benchmarks/data/spdata/";
    exec(generateData.c_str());
    std::string filename = "" + path_to_artifact + "/mosaic-benchmarks/data/spdata/spmv/B_" + std::to_string(dim) + "_0.2.mtx";
    B = castToType<float>("B", readMTX(filename, CSR));

    for (int i = 0; i < dim; i++) {
      C.insert({i}, (float)i);
    }
        
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
    A.compile(stmt);
    A.assemble();
    auto func = A.compute_split();
    auto pair = A.returnFuncPackedRaw(func);
    state.ResumeTiming();
    pair.first(func.data());
  }
  std::string eraseData = "rm -rf " + path_to_artifact + "/mosaic-benchmarks/data/spdata/spmv/B_" + std::to_string(dim) + "_0.2.mtx";
  exec(eraseData.c_str());
}

TACO_BENCH(bench_spmv_taco)->DenseRange(250, 5000, 250);

