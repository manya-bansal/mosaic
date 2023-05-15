
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
#include "codegen/codegen.h"
#include "taco/lower/lowerer_impl_imperative.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/lower/lower.h"
#include "taco/ir_tags.h"
#include "taco/error/error_messages.h"

#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_interface/test_interface.h"
#include "taco/accelerator_interface/cuda_interface.h"
#include "taco/accelerator_interface/tile_interface.h"
#include "taco/accelerator_interface/tensorflow_interface.h"
#include "taco/accelerator_interface/gsl_interface.h"
#include "taco/accelerator_interface/tensor_interface.h"
#include "taco/accelerator_interface/avx2_interface.h"
#include "taco/accelerator_interface/dynamic_order_interface.h"
#include "taco/accelerator_interface/mkl_interface.h"
#include "taco/accelerator_interface/stardust_interface.h"


using namespace taco;

extern bool gsl_compile;
extern bool mkl_compile;

TEST(autoschedule, endToEndPluginInterfaceClass) {

   // actual computation
   Tensor<float> A("A", {16}, Format{Dense});
   Tensor<float> B("B", {16}, Format{Dense});
   Tensor<float> C("C", {16}, Format{Dense});
   IndexVar i("i");
   

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

   C.pack();
   B.pack();

   A(i) = B(i) + C(i) + B(i);

   // register the description
   A.registerAccelerator(new Saxpy());
   // enable targeting
   A.accelerateOn();
   
   A.compile();
   A.assemble();
   A.compute();

   Tensor<float> expected("expected", {16}, Format{Dense});
   expected(i) = B(i) + C(i) + B(i);
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);

}

TEST(autoschedule, endToEndSdot) {

   // actual computation
   Tensor<float> A("A", {16}, Format{Dense});
   Tensor<float> B("B", {16}, Format{Dense});
   Tensor<float> C("C", {16}, Format{Dense});
   IndexVar i("i");

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

   C.pack();
   B.pack();

   A(i) = sum(i, B(i) * C(i)) + B(i);

   // register the description
   A.registerAccelerator(new Sdot());
   // enable targeting
   A.accelerateOn();
   
   A.compile();
   A.assemble();
   A.compute();

   Tensor<float> expected("expected", {16}, Format{Dense});
   expected(i) = sum(i, B(i) * C(i)) + B(i);
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);

}

TEST(autoschedule, cblasSgmev) {

   // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> b("b", {16}, Format{Dense});
   Tensor<float> c("c", {16}, Format{Dense});
   Tensor<float> d("d", {16}, Format{Dense});

   for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
         A.insert({i, j}, (float) i + j);
      }
   }

   A.pack();

   for (int i = 0; i < 16; i++) {
      c.insert({i}, (float) i);
      b.insert({i}, (float) i);
   }

   c.pack();
   b.pack();

   Tensor<float> expected("expected", {16}, Format{Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = A(i, j) * b(j) + c(i);
   d(i) = accelerateExpr;

   // register the description
   d.registerAccelerator(new Sgemv());
   // enable targeting
   d.accelerateOn();
   
   d.compile();
   d.assemble();
   d.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, d);

}

TEST(autoschedule, cblasSgemm) {

   // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> B("B", {16, 16}, Format{Dense, Dense});
   Tensor<float> C("C", {16, 16}, Format{Dense, Dense});
   Tensor<float> D("D", {16, 16}, Format{Dense, Dense});

   Tensor<float> expected("expected", {16, 16}, Format{Dense, Dense});

   for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
         C.insert({i, j}, (float) i + j*i);
         B.insert({i, j}, (float) i + j*i*4);
         D.insert({i, j}, (float) i + j*i*6);
      }
   }

   C.pack();
   B.pack();
   D.pack();


   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = B(i, j) * C(j, k) + D(i,k);
   A(i, k) = accelerateExpr;

   // register the description
   A.registerAccelerator(new Sgemm());
   // enable targeting
   A.accelerateOn();
   
   A.compile();
   A.assemble();
   A.compute();

   expected(i, k) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);

}

TEST(autoschedule, tblisSgemm) {

   // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> B("B", {16, 16}, Format{Dense, Dense});
   Tensor<float> C("C", {16, 16}, Format{Dense, Dense});
   Tensor<float> D("D", {16, 16}, Format{Dense, Dense});

   Tensor<float> expected("expected", {16, 16}, Format{Dense, Dense});

   for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
         C.insert({i, j}, (float) i + j);
         B.insert({i, j}, (float) i + j);
      }
   }

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = B(i, j) * C(j, k) + C(i,k);
   A(i, k) = accelerateExpr;


   // IndexStmt stmt = A.getAssignment().concretize();
   // stmt = stmt.accelerate(new TblisMultiply(), accelerateExpr);
   A.registerAccelerator(new TblisMultiply());
   // enable targeting
   A.accelerateOn();
   
   A.compile();
   A.assemble();
   A.compute();

   expected(i, k) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);

}

TEST(autoschedule, tblisSaxpyUnfused) {


   Tensor<float> A("A", {16}, Format{Dense}, 0);
   Tensor<float> B("B", {16}, Format{Dense});
   Tensor<float> C("C", {16}, Format{Dense});
   Tensor<float> expected("expected", {16}, Format{Dense});
   TensorVar accelWorkspace("accelWorkspace", Type(taco::Float32, {16}), taco::dense);
   IndexVar i("i");
   IndexVar iw("iw");

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

   C.pack();
   B.pack();

   IndexExpr accelerateExpr = B(i) + C(i) + B(i);
   A(i) = accelerateExpr;

   A.registerAccelerator(new TblisSaxpy());
   A.accelerateOn();
   
   A.compile();
   A.assemble();
   A.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);
}

TEST(autoschedule, gslSgmev) {

   gsl_compile = true;

   // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> b("b", {16}, Format{Dense});
   Tensor<float> c("c", {16}, Format{Dense});
   Tensor<float> d("d", {16}, Format{Dense});

   for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
         A.insert({i, j}, (float) i + j);
      }
   }

   A.pack();

   for (int i = 0; i < 16; i++) {
      c.insert({i}, (float) i);
      b.insert({i}, (float) i);
   }

   c.pack();
   b.pack();

   Tensor<float> expected("expected", {16}, Format{Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = A(i, j) * b(j) + c(i);
   d(i) = accelerateExpr;

   // register the description
   d.registerAccelerator(new GSLSgemv());
   // enable targeting
   d.accelerateOn();
   
   d.compile();
   d.assemble();
   d.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, d);

   gsl_compile = false;
}