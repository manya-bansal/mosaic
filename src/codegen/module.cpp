#include "taco/codegen/module.h"

#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <unistd.h>
#if USE_OPENMP
#include <omp.h>
#endif

#include "taco/tensor.h"
#include "taco/error.h"
#include "taco/util/strings.h"
#include "taco/util/env.h"
#include "codegen/codegen_c.h"
#include "codegen/codegen_cuda.h"
#include "taco/cuda.h"

using namespace std;

bool gsl_compile = false;
bool mkl_compile = false;

namespace taco {
namespace ir {

std::string Module::chars = "abcdefghijkmnpqrstuvwxyz0123456789";
std::default_random_engine Module::gen = std::default_random_engine();
std::uniform_int_distribution<int> Module::randint =
    std::uniform_int_distribution<int>(0, chars.length() - 1);

void Module::setJITTmpdir() {
  tmpdir = util::getTmpdir();
  // cout << tmpdir << endl;
}

void Module::setJITLibname() {
  libname.resize(12);
  for (int i=0; i<12; i++)
    libname[i] = chars[randint(gen)];
}

void Module::addFunction(Stmt func) {
  funcs.push_back(func);
}

void Module::compileToSource(string path, string prefix) {
  if (!moduleFromUserSource) {
  
    // create a codegen instance and add all the funcs
    bool didGenRuntime = false;
    
    header.str("");
    header.clear();
    source.str("");
    source.clear();

    taco_tassert(target.arch == Target::C99) <<
        "Only C99 codegen supported currently";
    std::shared_ptr<CodeGen> sourcegen =
        CodeGen::init_default(source, CodeGen::ImplementationGen);
    std::shared_ptr<CodeGen> headergen =
            CodeGen::init_default(header, CodeGen::HeaderGen);

    for (auto func: funcs) {
      sourcegen->compile(func, !didGenRuntime);
      headergen->compile(func, !didGenRuntime);
      didGenRuntime = true;
    }
  }

  ofstream source_file;
  string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";

  source_file.open(path+prefix+file_ending);
  source_file << source.str();
  source_file.close();
  
  ofstream header_file;
  header_file.open(path+prefix+".h");
  header_file << header.str();
  header_file.close();

}

void Module::compileToStaticLibrary(string path, string prefix) {
  taco_tassert(false) << "Compiling to a static library is not supported";
}
  
namespace {

void writeShims(vector<Stmt> funcs, string path, string prefix) {
  stringstream shims;
  for (auto func: funcs) {
    if (should_use_CUDA_codegen()) {
      CodeGen_CUDA::generateShim(func, shims);
    }
    else {
      CodeGen_C::generateShim(func, shims);
    }
  }
  
  ofstream shims_file;
  if (should_use_CUDA_codegen()) {
    shims_file.open(path+prefix+"_shims.cpp");
  }
  else {
    shims_file.open(path+prefix+".c", ios::app);
  }
  shims_file << "#include \"" << path << prefix << ".h\"\n";
  shims_file << shims.str();
  shims_file.close();
}

} // anonymous namespace

string Module::compile() {
  string prefix = tmpdir+libname;
  string fullpath = prefix + ".so";
  
  string cc;
  string cflags;
  string file_ending;
  string shims_file;
  if (should_use_CUDA_codegen()) {
    cc = util::getFromEnv("TACO_NVCC", "nvcc");
    cflags = util::getFromEnv("TACO_NVCCFLAGS",
    get_default_CUDA_compiler_flags());
    file_ending = ".cu";
    shims_file = prefix + "_shims.cpp";
  }
  else {
    cc = util::getFromEnv(target.compiler_env, target.compiler);
#ifdef TACO_DEBUG
    // In debug mode, compile the generated code with debug symbols and a
    // low optimization level.
    string defaultFlags = "-g -O0 -std=c99";
#else
    // Otherwise, use the standard set of optimizing flags.
    string defaultFlags = "-std=c99";
#endif
    cflags = util::getFromEnv("TACO_CFLAGS", defaultFlags) + " -shared -fPIC";
#if USE_OPENMP
    cflags += " -fopenmp";
#endif
    file_ending = ".c";
    shims_file = "";
  }

  
  string cmd = cc + " " + cflags + " " +
    prefix + file_ending + " " + shims_file + " " + 
    "-o " + fullpath + " -lm";



  // GSL has its own implementatio of cblas, 
  // we need to include the correct one depending on what 
  // we are compiling 
  if (gsl_compile){
    cmd += " -fPIC"
           " -I/home/manya227/mosaic-artifact/tensor_algebra_systems_lib"
           " -I/home/manya227/tensor-algebra-systems/tblis/include"
           " -I/home/manya227/spblas_0_8"
           " -I/home/manya227/tensor-algebra-systems/tblis/include/tblis"
           " -I/home/manya227/tensor-algebra-systems/gsl/include" 
           " -I/opt/intel/compilers_and_libraries/linux/mkl/include"
           " -I/home/manya227/tensor-algebra-systems/cblas"
           " -I/home/manya227/tensor-algebra-systems/tensor-gsl/include/tensor"
           " -L/home/manya227/tensor-algebra-systems/tblis/lib"
           " -L/home/manya227/tensor-algebra-systems/gsl/lib"
           " -L/home/manya227/tensor-algebra-systems/tensor-gsl/lib"
           " /home/manya227/spblas_0_8/libsparseblas.a"
           " -L/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/"
           " -Wl,-R/home/manya227/tensor-algebra-systems/tblis/lib -l:libtblis.so.0.0.0 "
           " -Wl,-R/home/manya227/tensor-algebra-systems/gsl/lib -l:libgsl.so.25.1.0"
           " -Wl,-R/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/ -lmkl_intel_thread -lmkl_rt -lmkl_core -l:libmkl_intel_lp64.so"
          // "  -lmkl_sequential -lmkl_avx512 -lmkl_core -lpthread -lm -ldl -lmkl_avx2 -lmkl_def -lmkl_rt"
           " -Wl,-R/home/manya227/tensor-algebra-systems/tensor-gsl/lib -l:libtensor.so.0.0.0 -l:libgslcblas.so.0.0.0";
          
  }
  else {
    cmd += " -mavx2 -fPIC"
           " -I/home/manya227/mosaic-artifact/tensor_algebra_systems_lib"
           " -I/home/manya227/tensor-algebra-systems/tblis/include"
           " -I/home/manya227/spblas_0_8"
           " -I/home/manya227/tensor-algebra-systems/tblis/include/tblis"
           " -I/home/manya227/tensor-algebra-systems/cblas"
           " -I/opt/intel/compilers_and_libraries/linux/mkl/include"
           " -I/home/manya227/tensor-algebra-systems/gsl/include" 
           " -I/home/manya227/tensor-algebra-systems/tensor-gsl/include/tensor"
           " -I/home/manya227/tensor-algebra-systems/" 
           " -L/home/manya227/tensor-algebra-systems/cblas"
           " -L/home/manya227/tensor-algebra-systems/tblis/lib"
           " -L/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/"
           " /home/manya227/spblas_0_8/libsparseblas.a"
           " -L/home/manya227/tensor-algebra-systems/gsl/lib"
           " -Wl,-R/home/manya227/tensor-algebra-systems/cblas -l:libblas.so.3.7.1" 
           " -Wl,-R/home/manya227/tensor-algebra-systems/tblis/lib -l:libtblis.so.0.0.0 "
           " -Wl,-R/home/manya227/tensor-algebra-systems/gsl/lib -l:libgsl.so.25.1.0"
          //  " -Wl,-R/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/ -l:libmkl_intel_lp64.so"
            " -Wl,-R/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/ -lmkl_intel_thread -lmkl_rt -lmkl_core -l:libmkl_intel_lp64.so";
          //  " -lmkl_sequential -lmkl_avx512 -lmkl_core -lpthread -lm -ldl -lmkl_avx2 -lmkl_def -lmkl_rt";
  }


  // cmd += " -lblas";
    
  // std::cout << cmd << std::endl;
  // open the output file & write out the source

  compileToSource(tmpdir, libname);
  
  // write out the shims
  writeShims(funcs, tmpdir, libname);
  
  // now compile it
  int err = system(cmd.data());
  taco_uassert(err == 0) << "Compilation command failed:\n" << cmd
    << "\nreturned " << err;

  // use dlsym() to open the compiled library
  if (lib_handle) {
    dlclose(lib_handle);
  }
  lib_handle = dlopen(fullpath.data(), RTLD_NOW | RTLD_LOCAL);
  taco_uassert(lib_handle) << "Failed to load generated code, error is: " << dlerror();

  return fullpath;
}

void Module::setSource(string source) {
  this->source << source;
  moduleFromUserSource = true;
}

string Module::getSource() {
  return source.str();
}

void* Module::getFuncPtr(std::string name) {
  return dlsym(lib_handle, name.data());
}

int Module::callFuncPackedRaw(std::string name, void** args) {
  typedef int (*fnptr_t)(void**);
  static_assert(sizeof(void*) == sizeof(fnptr_t),
    "Unable to cast dlsym() returned void pointer to function pointer");
  void* v_func_ptr = getFuncPtr(name);
  fnptr_t func_ptr;
  *reinterpret_cast<void**>(&func_ptr) = v_func_ptr;

#if USE_OPENMP
  omp_sched_t existingSched;
  ParallelSchedule tacoSched;
  int existingChunkSize, tacoChunkSize;
  int existingNumThreads = omp_get_max_threads();
  omp_get_schedule(&existingSched, &existingChunkSize);
  taco_get_parallel_schedule(&tacoSched, &tacoChunkSize);
  switch (tacoSched) {
    case ParallelSchedule::Static:
      omp_set_schedule(omp_sched_static, tacoChunkSize);
      break;
    case ParallelSchedule::Dynamic:
      omp_set_schedule(omp_sched_dynamic, tacoChunkSize);
      break;
    default:
      break;
  }
  omp_set_num_threads(taco_get_num_threads());
#endif

  int ret = func_ptr(args);

#if USE_OPENMP
  omp_set_schedule(existingSched, existingChunkSize);
  omp_set_num_threads(existingNumThreads);
#endif

  return ret;
}

std::pair<int (*)(void**),void**>  Module::returnFuncPackedRaw(std::string name, void** args) {
  typedef int (*fnptr_t)(void**);
  static_assert(sizeof(void*) == sizeof(fnptr_t),
    "Unable to cast dlsym() returned void pointer to function pointer");
  void* v_func_ptr = getFuncPtr(name);
  fnptr_t func_ptr;
  *reinterpret_cast<void**>(&func_ptr) = v_func_ptr;

#if USE_OPENMP
  omp_sched_t existingSched;
  ParallelSchedule tacoSched;
  int existingChunkSize, tacoChunkSize;
  int existingNumThreads = omp_get_max_threads();
  omp_get_schedule(&existingSched, &existingChunkSize);
  taco_get_parallel_schedule(&tacoSched, &tacoChunkSize);
  switch (tacoSched) {
    case ParallelSchedule::Static:
      omp_set_schedule(omp_sched_static, tacoChunkSize);
      break;
    case ParallelSchedule::Dynamic:
      omp_set_schedule(omp_sched_dynamic, tacoChunkSize);
      break;
    default:
      break;
  }
  omp_set_num_threads(taco_get_num_threads());
#endif
  return {func_ptr, args};
}
 // namespace ir
} // namespace taco

}
