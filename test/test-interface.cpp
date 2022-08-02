#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include <taco/index_notation/transformations.h>
#include <codegen/codegen_c.h>
#include <codegen/codegen_cuda.h>
#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/accel_interface.h"
#include "codegen/codegen.h"
#include "taco/lower/lowerer_impl_imperative.h"
#include "taco/index_notation/accelerate_notation.h"
#include "taco/lower/lower.h"
#include "taco/ir_tags.h"


using namespace taco;


bool trivialkernelChecker(IndexStmt expr){
   return true;
}


TEST(transferType, pluginInterface) {

   TensorVar a("a", Type(taco::Float32, {Dimension()}), taco::dense);
   TensorVar b("b", Type(taco::Float32, {Dimension()}), taco::dense);
   IndexVar i("i");

   // should basically call a C function 
   // that can be included in header
   TransferLoad load_test("load_test", "void");
   TransferStore store_test("store_test", "void");

   TransferType kernelTransfer("test", load_test, store_test);

   ForeignFunctionDescription kernel1("kernel1", "void", a(i),  a(i) + b(i), {}, trivialkernelChecker);
   ForeignFunctionDescription kernel2( "kernel2", "void", a(i), b(i), {}, trivialkernelChecker);

   AcceleratorDescription accelDesc(kernelTransfer, 
            {  kernel1(load_test(a)),
               kernel2(load_test(a, load_test(b)), load_test(b))
            });

   cout << load_test(a, load_test(a), load_test(a, load_test(a)), b, Dim(i)) << endl;

   Tensor<double> A("A", {16}, Format{Dense});

    //need to register AcceleratorDescription
    //so that the TACO can use it

}


TEST(transferType, concretepluginInterface) {

  Tensor<float> A("A", {16}, Format{Dense}, 0);
  Tensor<float> B("B", {16}, Format{Dense});
  Tensor<float> C("C", {16}, Format{Dense});
  TensorVar a("a", Type(taco::Float32, {Dimension()}), taco::dense);

  IndexVar i("i");

  for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
  }

  C.pack();
  B.pack();

   // should basically call a C function 
   // that can be included in header
   TransferLoad load_test("load_test", "void");
   TransferStore store_test("store_test", "void");

   TransferType kernelTransfer("test", load_test, store_test);

   ForeignFunctionDescription cblas_saxpy("cblas_saxpy", "void", C(i),  A(i) + B(i), {}, trivialkernelChecker);

   AcceleratorDescription accelDesc(kernelTransfer, 
            {  cblas_saxpy(Dim(i), A),
            });
   
   cout << cblas_saxpy(Dim(i), 1, A, 1, B, 1) << endl;


   ConcreteAccelerateCodeGenerator concrete_cblas_saxpy("cblas_saxpy", "void",  A(i),  A(i) + B(i), {});

   cout << concrete_cblas_saxpy(Dim(i), 1, A, 1, B, 1) << endl;

    //need to register AcceleratorDescription
    //so that the TACO can use it

}