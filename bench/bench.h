#ifndef TACO_BENCH_BENCH_H
#define TACO_BENCH_BENCH_H

#include "benchmark/benchmark.h"
#include "taco/tensor.h"

// Register a benchmark with the following options:
// * Millisecond output display
// * 10 data points
// * Reporting of avg/stddev/median
// * Wall-clock time, rather than CPU time.
#define TACO_BENCH(bench)         \
  BENCHMARK(bench)                \
  ->Unit(benchmark::kMillisecond) \
  ->Repetitions(10)               \
  ->Iterations(1)                 \
  ->ReportAggregatesOnly(true)    \
  ->UseRealTime()                 \
  ->MinWarmUpTime(10)

#define GRAPHBLAS_BENCH(bench,times)   \
  BENCHMARK(bench)                \
  ->Unit(benchmark::kMillisecond) \
  ->Repetitions(times)               \
  ->Iterations(1)                 \
  ->ReportAggregatesOnly(false)    \
  ->UseRealTime()

// TACO_BENCH_ARG is similar to TACO_BENCH but allows for passing
// of an arbitrarily typed argument to the benchmark function.
// TODO (rohany): Make this take in only 1 argument.
// TODO (rohany): Don't specify the time here, but do it at the command line.

#define TACO_BENCH_ARG(bench, name, arg)  \
  BENCHMARK_CAPTURE(bench, name, arg)     \
  ->Unit(benchmark::kMillisecond)         \
  ->Repetitions(10)                       \
  ->Iterations(1)                         \
  ->ReportAggregatesOnly(true)            \
  ->UseRealTime()                         \
  ->MinWarmUpTime(10)

#define TACO_BENCH_ARGS(bench, name, ...)       \
  BENCHMARK_CAPTURE(bench, name, __VA_ARGS__)   \
  ->Unit(benchmark::kMicrosecond)               \
  ->Iterations(1)                               \
  ->ReportAggregatesOnly(true)                  \
  ->UseRealTime()

std::string getEnvVar(std::string varname);
std::string getTacoTensorPath();
std::string getValidationOutputPath();
// cleanPath ensures that the input path ends with "/".
std::string cleanPath(std::string path);
taco::TensorBase loadRandomTensor(std::string name, std::vector<int> dims, float sparsity, taco::Format format, int variant=0);
taco::TensorBase loadImageTensor(std::string name, int num, taco::Format format, float threshold, int variant=0);
taco::TensorBase loadMinMaxTensor(std::string name, int order, taco::Format format, int variant=0);
template<typename T>
taco::Tensor<T> castToType(std::string name, taco::Tensor<double> tensor) {
  taco::Tensor<T> result(name, tensor.getDimensions(), tensor.getFormat());
  std::vector<int> coords(tensor.getOrder());
  for (auto& value : taco::iterate<double>(tensor)) {
    for (int i = 0; i < tensor.getOrder(); i++) {
      coords[i] = value.first[i];
    }
    // Attempt to cast the value to an integer. However, if the cast causes
    // the value to equal 0, then this will ruin the sparsity pattern of the
    // tensor, as the 0 values will get compressed out. So, if a cast would
    // equal 0, insert 1 instead to preserve the sparsity pattern of the tensor.
    if (static_cast<T>(value.second) == T(0)) {
      result.insert(coords, static_cast<T>(1));
    } else {
      result.insert(coords, static_cast<T>(value.second));
    }
  }
  result.pack();
  return result;
}

template<typename T>
taco::Tensor<T> castToTypeZero(std::string name, taco::Tensor<double> tensor) {
  taco::Tensor<T> result(name, tensor.getDimensions(), tensor.getFormat());
  std::vector<int> coords(tensor.getOrder());
  for (auto& value : taco::iterate<double>(tensor)) {
    for (int i = 0; i < tensor.getOrder(); i++) {
      coords[i] = value.first[i];
    }
    // Attempt to cast the value to an integer. However, if the cast causes
    // the value to equal 0, then this will ruin the sparsity pattern of the
    // tensor, as the 0 values will get compressed out. So, if a cast would
    // equal 0, insert 1 instead to preserve the sparsity pattern of the tensor.
    result.insert(coords, static_cast<T>(value.second));
  }
  result.pack();
  return result;
}

template<typename T, typename T2>
taco::Tensor<T> shiftLastMode(std::string name, taco::Tensor<T2> original) {
  taco::Tensor<T> result(name, original.getDimensions(), original.getFormat());
  std::vector<int> coords(original.getOrder());
  for (auto& value : taco::iterate<T2>(original)) {
    for (int i = 0; i < original.getOrder(); i++) {
      coords[i] = value.first[i];
    }
    int lastMode = original.getOrder() - 1;
    // For order 2 tensors, always shift the last coordinate. Otherwise, shift only coordinates
    // that have even last coordinates. This ensures that there is at least some overlap
    // between the original tensor and its shifted counter part.
    if (original.getOrder() <= 2 || (coords[lastMode] % 2 == 0)) {
      coords[lastMode] = (coords[lastMode] + 1) % original.getDimension(lastMode);
    }
    // TODO (rohany): Temporarily use a constant value here.
    result.insert(coords, T(2));
  }
  result.pack();
  return result;
}

#endif //TACO_BENCH_BENCH_H
