#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include <Accelerate/Accelerate.h>
#include <taco/index_notation/transformations.h>
#include <codegen/codegen_c.h>
#include <codegen/codegen_cuda.h>
#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "codegen/codegen.h"
#include "taco/lower/lowerer_impl_imperative.h"
#include "taco/index_notation/accel_interface.h"
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

   Tensor<float32_t> A("A", {16}, Format{Dense}, 0);
   Tensor<float32_t> B("B", {16}, Format{Dense});
   Tensor<float32_t> C("C", {16}, Format{Dense});

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float32_t) i);
      B.insert({i}, (float32_t) i);
   }

   C.pack();
   B.pack();

   IndexVar i("i");
   IndexVar iw("iw");
   IndexExpr accelerateExpr = B(i) + C(i);
   A(i) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();


   //the way rewrite works, you need an object to object copy 
   ConcreteAccelerateCodeGenerator concrete_cblas_saxpy("cblas_saxpy", "void",  B(i), accelerateExpr, {});
   cout << concrete_cblas_saxpy(Dim(i), 1, A, 1, B, 1) << endl;

   TensorVar accelWorkspace("accelWorkspace", Type(taco::Float32, {16}), taco::dense);

   stmt = stmt.accelerate(concrete_cblas_saxpy(Dim(i), 1, A, 1, B, 1), i, iw, accelWorkspace);
   

   //need to register AcceleratorDescription
   //so that the TACO can use it

}