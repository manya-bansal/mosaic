#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"



#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_interface/gsl_interface.h"
#include "taco/accelerator_interface/tensor_interface.h"

#include "taco/storage/file_io_mtx.h"

#define DIM_SIZE 3000

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

static void bench_spmv_double_sp_gsl(benchmark::State& state, float SPARSITY, int dim) {
   
    Tensor<double> B("B", {dim, dim}, CSR);
    Tensor<double> C("C", {dim}, Format{Dense});

    char const *ret = getenv("PATH_TO_MOSAIC_ARTIFACT");
    if (!ret) {
      taco_uerror << "Please set the environment variable PATH_TO_MOSAIC_ARTIFACT."
      << "To do so, run (in the mosaic/bench/bench-scripts/ dir): source mosaic_env_var.sh.";
    }

    std::string path_to_artifact = std::string(ret);

    std::string generateData = "python3  " + path_to_artifact + "/mosaic-benchmarks/data/data_gen.py --bench spmv --dim ";
    generateData += std::to_string(dim);

    generateData += " --nnz ";
    std::ostringstream ss;
    ss << SPARSITY;generateData += ss.str();
    generateData += " --out_dir " + path_to_artifact + "/mosaic-benchmarks/data/spdata/";

    exec(generateData.c_str());
    std::string filename = path_to_artifact + "/mosaic-benchmarks/data/spdata/spmv/B_"+ std::to_string(dim) + "_" + ss.str() + ".mtx";
    B = castToType<double>("B", readMTX(filename, CSR));

    for (int i = 0; i < dim; i++) {
      C.insert({i}, (double)i);
    }

    IndexVar i("i");
    IndexVar j("j");
    IndexVar k("k");

    IndexExpr accelerateExpr = B(i, j) * C(j);
  
   for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<double> A("A", {dim}, Format{Dense});
    A(i) = B(i, j) * C(j);
    IndexStmt stmt = A.getAssignment().concretize();
    stmt = stmt.accelerate(new GslSpMV(), accelerateExpr, true);
    A.compile(stmt);
    A.assemble();
    auto func = A.compute_split();
    auto pair = A.returnFuncPackedRaw(func);
    state.ResumeTiming();
    pair.first(func.data());
  }
  std::string eraseData = "rm -rf " + path_to_artifact + "/mosaic-benchmarks/data/spdata/spmv/B_" + std::to_string(dim) + "_" + ss.str() + ".mtx";
  exec(eraseData.c_str());
}

TACO_BENCH_ARGS(bench_spmv_double_sp_gsl, 0.00078125, 0.00078125, DIM_SIZE);
TACO_BENCH_ARGS(bench_spmv_double_sp_gsl, 0.0015625, 0.0015625, DIM_SIZE);
TACO_BENCH_ARGS(bench_spmv_double_sp_gsl, 0.003125, 0.003125, DIM_SIZE);
TACO_BENCH_ARGS(bench_spmv_double_sp_gsl, 0.00625, 0.00625, DIM_SIZE);
TACO_BENCH_ARGS(bench_spmv_double_sp_gsl, 0.0125,  0.0125, DIM_SIZE);
TACO_BENCH_ARGS(bench_spmv_double_sp_gsl, 0.025,   0.025, DIM_SIZE);
TACO_BENCH_ARGS(bench_spmv_double_sp_gsl, 0.05,    0.05, DIM_SIZE);
TACO_BENCH_ARGS(bench_spmv_double_sp_gsl, 0.1,     0.1, DIM_SIZE);