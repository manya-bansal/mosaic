cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
project(taco-bench)
set(CMAKE_BUILD_TYPE "release")
file(GLOB TEST_SOURCES *.cpp *.h)

add_subdirectory(benchmark)
# add_subdirectory(${TACO_PROJECT_DIR}/benchmark/include)
include_directories(benchmark)



set(CMAKE_CXX_FLAGS "${C_CXX_FLAGS} -std=c++14")

add_executable(taco-bench ${TEST_SOURCES} bench.h)
target_link_libraries(taco-bench benchmark::benchmark)
target_link_libraries(taco-bench taco)
target_link_libraries(taco-bench benchmark_main)


