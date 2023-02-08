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


using namespace taco;

extern bool gsl_compile;
extern bool mkl_compile;

bool trivialkernelChecker(IndexStmt expr){
   return true;
}


TEST(interface, pluginInterface) {

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

   Tensor<float> A("A", {16}, Format{Dense});

}


TEST(interface, endToEndPlugin) {

   TensorVar x("x", Type(taco::Float32, {Dimension()}), taco::dense);
   TensorVar y("y", Type(taco::Float32, {Dimension()}), taco::dense);
   IndexVar i("i");

   ForeignFunctionDescription cblas_saxpy("cblas_saxpy", "void", x(i) <=  x(i) + y(i), {}, trivialkernelChecker);

   AcceleratorDescription accelDesc({cblas_saxpy(Dim(i), 1, y, 1, x, 1)});

   // actual computation
   Tensor<float> A("A", {16}, Format{Dense});
   Tensor<float> B("B", {16}, Format{Dense});
   Tensor<float> C("C", {16}, Format{Dense});

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

   C.pack();
   B.pack();

   A(i) = B(i) + C(i);

   // register the description
   // A.registerAccelerator(accelDesc);
   // enable targeting
   // A.accelerateOn();
   
   A.compile();
   A.assemble();
   A.compute();

   Tensor<float> expected("expected", {16}, Format{Dense});
   expected(i) = B(i) + C(i);
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);

}

TEST(interface, interfaceClass1) {


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

   IndexExpr accelerateExpr = B(i) + C(i);
   A(i) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new Saxpy(), accelerateExpr);

   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);
}

TEST(interface, interfaceClass2) {


   Tensor<float> A("A", {16}, Format{Dense}, 0);
   Tensor<float> B("B", {16}, Format{Dense});
   Tensor<float> C("C", {16}, Format{Dense});
   Tensor<float> expected("expected", {16}, Format{Dense}, 0);
   TensorVar accelWorkspace("accelWorkspace", Type(taco::Float32, {16}), taco::dense);
   IndexVar i("i");
   IndexVar j("j");
   IndexVar iw("iw");

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

   C.pack();
   B.pack();

   IndexExpr accelerateExpr = B(j) * C(j);
   A(i) = sum(j, accelerateExpr);

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new Sdot(), accelerateExpr);
    
   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i) = sum(j, accelerateExpr);
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);
}

TEST(interface, endToEndPluginInterfaceClass) {

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

TEST(interface, mismatchInterfaceClass) {


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

   IndexExpr accelerateExpr = B(i) + C(i);
   A(i) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();

   ASSERT_THROW(stmt.accelerate(new Sdot(), accelerateExpr), taco::TacoException);

}


TEST(interface, endToEndSdot) {

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

TEST(DISABLED_interface, endToEndUserDefinedDummy) {

   // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> B("B", {16, 16}, Format{Dense, Dense});
   Tensor<float> C("C", {16, 16}, Format{Dense, Dense});
   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = B(i, j) * C(j, k);
   A(i, k) = accelerateExpr;


   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new TestInterface(), accelerateExpr);
   
   A.compile(stmt);
   A.assemble();
   A.compute();

}

TEST(interface, endToEndUserDefinedErrorDummy) {

   // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> B("B", {16, 16}, Format{Dense, Dense});
   Tensor<float> C("C", {16, 16}, Format{Dense, Dense});
   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = B(i, j) * C(j, k);
   A(i, k) = accelerateExpr;


   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new TestInterfaceIncorrect(), accelerateExpr);

   ASSERT_THROW(A.compile(stmt), taco::TacoException);

}

TEST(interface, tiledSaxpyInterface) {


   Tensor<float> A("A", {16}, Format{Dense});
   Tensor<float> expected("expected", {16}, Format{Dense});
   Tensor<float> B("B", {16}, Format{Dense});
   Tensor<float> C("C", {16}, Format{Dense});
   IndexVar i("i");

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

   C.pack();
   B.pack();

   IndexExpr accelerateExpr = B(i) + C(i);
   A(i) = accelerateExpr;
   IndexStmt stmt = A.getAssignment().concretize();
   
   stmt = stmt.tile(new TileSaxpy(), accelerateExpr, {{i,  4}});

   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);
}


TEST(interface, tiledMMInterface) {

     // actual computation
   Tensor<float> A("A", {4, 4}, Format{Dense, Dense});
   Tensor<float> B("B", {4, 4}, Format{Dense, Dense});
   Tensor<float> C("C", {4, 4}, Format{Dense, Dense});
   Tensor<float> expected("expected", {4, 4}, Format{Dense, Dense});

   for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
         C.insert({i, j}, (float) i+j);
         B.insert({i, j}, (float) i+j);
      }
   }

   B.pack();
   C.pack();

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = B(i, j) * C(j, k);
   A(i, k) = accelerateExpr;


   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.tile(new MatrixMultiply(), accelerateExpr, {{i, 2}, {j, 2}, {k, 2}});
   
   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i, k) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);
}

TEST(interface, sampleSplitExample) {


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

   IndexExpr accelerateExpr = B(i) + C(i);
   A(i) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.split(i, IndexVar(), IndexVar(), 4);
   // taco_uerror << stmt << endl;
   A.compile(stmt);
   A.assemble();
   A.compute();
}

TEST(interface, sampleMatrixMultiplyExample) {


   Tensor<float> A("A", {16, 16}, Format{Dense, Dense}, 0);
   Tensor<float> B("B", {16, 16}, Format{Dense, Dense});
   Tensor<float> C("C", {16, 16}, Format{Dense, Dense});
   Tensor<float> expected("expected", {16}, Format{Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   A(i, k) = B(i, j) * C(j, k);

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.split(j, IndexVar(), IndexVar(), 4);
   // taco_uerror << stmt << endl;
   A.compile(stmt);
   A.assemble();
   A.compute();
}

TEST(interface, endToEndDeclVarIncorrect) {

   // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> B("B", {16, 16}, Format{Dense, Dense});
   Tensor<float> C("C", {16, 16}, Format{Dense, Dense});
   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = B(i, j) * C(j, k);
   A(i, k) = accelerateExpr;


   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new TestInterfaceDeclVarIncorrect(), accelerateExpr);
   
   ASSERT_THROW(A.compile(stmt), taco::TacoException);

}


TEST(DISABLED_interface, endToEndDeclVarCorrect) {

   // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> B("B", {16, 16}, Format{Dense, Dense});
   Tensor<float> C("C", {16, 16}, Format{Dense, Dense});
   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = B(i, j) * C(j, k);
   A(i, k) = accelerateExpr;


   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new TestInterfaceDeclVar(), accelerateExpr);
   
   A.compile(stmt);
   A.assemble();
   A.compute();

}

TEST(interface, tblisMultiply) {

   // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> B("B", {16, 16}, Format{Dense, Dense});
   Tensor<float> C("C", {16, 16}, Format{Dense, Dense});
    Tensor<float> expected("expected", {16, 16}, Format{Dense, Dense});
   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = B(i, j) * C(j, k);
   A(i, k) = accelerateExpr;


   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new TblisMultiply(), accelerateExpr);
   
   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i, k) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);

}

TEST(interface, dimReduceSaxpy) {

   Tensor<float> A("A", {16, 16}, Format{Dense, Dense}, 0);
   Tensor<float> B("B", {16, 16}, Format{Dense, Dense});
   Tensor<float> C("C", {16, 16}, Format{Dense, Dense});
   Tensor<float> expected("expected", {16, 16}, Format{Dense, Dense});

   TensorVar precomputed("precomputed", Type(taco::Float32, {16, 16}), Format{Dense, Dense});

   for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
         B.insert({i, j}, (float) i + j);
         C.insert({i, j}, (float) i + j);
      }
   }

   B.pack();
   C.pack();

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = B(i, j) + C(i, j);
   A(i, j) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.holdConstant(new Saxpy(), accelerateExpr, {i}, precomputed(i, j));
   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i, j) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);
}

TEST(interface, cblasSgmev) {

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

TEST(interface, cblasSgemm) {

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

TEST(interface, tblisTTM) {

   // actual computation
   Tensor<float> A("A", {2, 2, 2}, Format{Dense, Dense, Dense});
   Tensor<float> B("B", {2, 2, 2}, Format{Dense, Dense, Dense});
   Tensor<float> C("C", {2, 2}, Format{Dense, Dense});
   Tensor<float> expected("expected", {2, 2, 2}, Format{Dense, Dense, Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
   IndexVar l("l");

   for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
         C.insert({i, j}, (float) i + j);
      }
   }

   for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
         for (int k = 0; k < 2; k++) {
            B.insert({i, j, k}, (float) i + j + k);
         }
      }
   }

   C.pack();
   B.pack();

   IndexExpr accelerateExpr = B(i, j, l) * C(k, l);
   A(i, j, k) = accelerateExpr;


   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new TblisTTM(), accelerateExpr);
   
   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i, j, k) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);

}

TEST(interface, tblisDot) {


   Tensor<float> A("A", {16}, Format{Dense}, 0);
   Tensor<float> B("B", {16}, Format{Dense});
   Tensor<float> C("C", {16}, Format{Dense});
   Tensor<float> expected("expected", {16}, Format{Dense}, 0);
   TensorVar accelWorkspace("accelWorkspace", Type(taco::Float32, {16}), taco::dense);
   IndexVar i("i");
   IndexVar j("j");
   IndexVar iw("iw");

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

   C.pack();
   B.pack();

   IndexExpr accelerateExpr = B(j) * C(j);
   A(i) = sum(j, accelerateExpr);

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new TblisDot(), accelerateExpr);
    
   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i) = sum(j, accelerateExpr);
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);
}

TEST(interface, tblisSgemm) {

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

TEST(interface, tblisSaxpy) {


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

   IndexExpr accelerateExpr = B(i) + C(i);
   A(i) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new TblisSaxpy(), accelerateExpr);

   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);
}

TEST(interface, tblisSaxpyUnfused) {


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


TEST(interface, cblassMMultiply) {

   // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> B("B", {16, 16}, Format{Dense, Dense});
   Tensor<float> C("C", {16, 16}, Format{Dense, Dense});
    Tensor<float> expected("expected", {16, 16}, Format{Dense, Dense});
   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = B(i, j) * C(j, k);
   A(i, k) = accelerateExpr;


   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new MatrixMultiply(), accelerateExpr);
   
   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i, k) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);

}


TEST(DISABLED_interface, tensorFlowCompile) {

   // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> B("B", {16, 16}, Format{Dense, Dense});
   Tensor<float> C("C", {16, 16}, Format{Dense, Dense});
    Tensor<float> expected("expected", {16, 16}, Format{Dense, Dense});
   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = B(i, j) * C(j, k);
   A(i, k) = accelerateExpr;


   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new TestTF(), accelerateExpr);
   
   A.compile(stmt);
   A.assemble();
   A.compute();

}

TEST(interface, gslVecAdd) {

   gsl_compile = true;

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

   IndexExpr accelerateExpr = B(i) + C(i);
   A(i) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new GSLVecAdd(), accelerateExpr);

   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);

   gsl_compile = false;
}

TEST(interface, gslDot) {

   gsl_compile = true;

   Tensor<float> A("A", {16}, Format{Dense}, 0);
   Tensor<float> B("B", {16}, Format{Dense});
   Tensor<float> C("C", {16}, Format{Dense});
   Tensor<float> expected("expected", {16}, Format{Dense}, 0);
   TensorVar accelWorkspace("accelWorkspace", Type(taco::Float32, {16}), taco::dense);
   IndexVar i("i");
   IndexVar j("j");
   IndexVar iw("iw");

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

   C.pack();
   B.pack();

   IndexExpr accelerateExpr = B(j) * C(j);
   A(i) = sum(j, accelerateExpr);

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new GSLDot(), accelerateExpr);
    
   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i) = sum(j, accelerateExpr);
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);

   gsl_compile = false;
}


TEST(interface, gslSgmev) {

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


TEST(interface, gslSgemm) {

   gsl_compile = true;

    // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> B("B", {16, 16}, Format{Dense, Dense});
   Tensor<float> C("C", {16, 16}, Format{Dense, Dense});
    Tensor<float> expected("expected", {16, 16}, Format{Dense, Dense});
   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");


   for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
         B.insert({i, j}, (float) i + j);
         C.insert({i, j}, (float) i + j);
      }
   }

   B.pack();
   C.pack();

   IndexExpr accelerateExpr = B(i, j) * C(j, k);
   A(i, k) = accelerateExpr;


   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new GSLMM(), accelerateExpr);
   
   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i, k) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);

   gsl_compile = false;
}


TEST(DISABLED_interface, gslTensorPlus) {

   gsl_compile = true;

   int dim = 16;

   // actual computation
   Tensor<float> A("A", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> B("B", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> C("C", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> expected("expected", {dim, dim, dim}, Format{Dense, Dense, Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
   IndexVar l("l");

   for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
         for (int k = 0; k < dim; k++) {
            B.insert({i, j, k}, (float) i + j * k);
            C.insert({i, j, k}, (float) i + j * k);
         }
      }
   }

   C.pack();
   B.pack();

   IndexExpr accelerateExpr = B(i, j, k) + C(i, j, k);
   A(i, j, k) = accelerateExpr;


   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new GslTensorPlus(), accelerateExpr, true);
   
   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i, j, k) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);

   gsl_compile = false;

}


TEST(interface, tblisPlus) {
   
   int dim = 2;

   // actual computation
   Tensor<float> A("A", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> B("B", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> C("C", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> expected("expected", {dim, dim, dim}, Format{Dense, Dense, Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
   IndexVar l("l");

   for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
         for (int k = 0; k < dim; k++) {
            B.insert({i, j, k}, (float) i + j * k);
            C.insert({i, j, k}, (float) i + j * k);
         }
      }
   }

   C.pack();
   B.pack();

   IndexExpr accelerateExpr = B(i, j, k) + C(i, j, k);
   A(i, j, k) = accelerateExpr;

   
   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new TblisPlus(), accelerateExpr, true);
   A.compile(stmt);
   // taco_uerror << "stop";
   A.assemble();
   A.compute();

   expected(i, j, k) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);


}


TEST(interface, DimReduceDot) {

   Tensor<float> A("A", {16, 16}, Format{Dense, Dense}, 0);
   Tensor<float> B("B", {16, 16}, Format{Dense, Dense});
   Tensor<float> C("C", {16, 16}, Format{Dense, Dense});
   Tensor<float> expected("expected", {16, 16}, Format{Dense, Dense});

   TensorVar precomputed("precomputed", Type(taco::Float32, {16, 16}), Format{Dense, Dense});

   for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
         B.insert({i, j}, (float) i + j);
         C.insert({i, j}, (float) i + j);
      }
   }

   B.pack();
   C.pack();

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = B(i, j) * C(j, k);
   A(i, k) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.holdConstant(new TblisDot(), accelerateExpr, {i, k}, precomputed(i, k));
   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i, k) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);

}



TEST(interface, DimReduceMM) {

   int dim = 16;

   Tensor<float> A("A", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> B("B", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> C("C", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> expected("expected", {dim, dim, dim}, Format{Dense, Dense, Dense});

   TensorVar precomputed("precomputed", Type(taco::Float32, {16, 16, 16}), Format{Dense, Dense, Dense});

   for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
         for (int k = 0; k < dim; k++) {
            B.insert({i, j, k}, (float) i + j * k);
            C.insert({i, j, k}, (float) i + j * k);
         }
      }
   }

   C.pack();
   B.pack();

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
   IndexVar l("l");

   IndexExpr accelerateExpr = B(i, j, l) * C(l, k, i);
   A(i, j, k) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.holdConstant(new MatrixMultiply(), accelerateExpr, {i}, precomputed(i, j, k));
   
   A.compile(stmt);
   A.assemble();
   A.compute();

}


TEST(DISABLED_interface, blockedSparseGSL) {

   gsl_compile = true;

   int dim = 16;

   Tensor<float> A("A", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> B("B", {dim, dim, dim}, Format{Sparse, Dense, Dense}); 
   Tensor<float> C("C", {dim, dim}, Format{Dense, Dense});

   float SPARSITY = .3;

   for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (float) ((int) (rand_float*3/SPARSITY)));
    }
   }


  for (int i = 0; i < dim; i++) {
   for (int j = 0; j < dim; j++) {
    for (int k = 0; k < dim; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, j, k}, (float) ((int) (rand_float*3/SPARSITY)));
      }
    }
   }
  }


   B.pack();
   C.pack();

   Tensor<float> expected("expected", {dim, dim, dim}, Format{Dense, Dense, Dense});

   TensorVar precomputed("precomputed", Type(taco::Float32, {Dimension(dim), Dimension(dim), Dimension(dim)}), Format{Dense, Dense, Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
   IndexVar l("l");
   IndexVar m("m");

   IndexExpr accelerateExpr = B(i, j, l) * C(l, k);
   A(i, j, k) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.holdConstant(new GSLMM(), accelerateExpr, {i}, precomputed(i, j, k));
   
   A.compile(stmt);
   A.assemble();
   A.compute();

   gsl_compile = false;

   expected(i, j, k) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);

   gsl_compile = false;


}

TEST(interface, blockedSparseCblas) {

   int dim = 16;

   Tensor<float> A("A", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> B("B", {dim, dim, dim}, Format{Sparse, Dense, Dense}); 
   Tensor<float> C("C", {dim, dim}, Format{Dense, Dense});

   float SPARSITY = .3;

   for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (float) ((int) (rand_float*3/SPARSITY)));
    }
   }


  for (int i = 0; i < dim; i++) {
   for (int j = 0; j < dim; j++) {
    for (int k = 0; k < dim; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, j, k}, (float) ((int) (rand_float*3/SPARSITY)));
      }
    }
   }
  }


   B.pack();
   C.pack();

   Tensor<float> expected("expected", {dim, dim, dim}, Format{Dense, Dense, Dense});

   TensorVar precomputed("precomputed", Type(taco::Float32, {Dimension(dim), Dimension(dim), Dimension(dim)}), Format{Dense, Dense, Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
   IndexVar l("l");
   IndexVar m("m");

   IndexExpr accelerateExpr = B(i, j, l) * C(l, k);
   A(i, j, k) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.holdConstant(new MatrixMultiply(), accelerateExpr, {i}, precomputed(i, j, k));
   
   A.compile(stmt);
   A.assemble();
   A.compute();

   gsl_compile = false;

   expected(i, j, k) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);
}

TEST(interface, blockedSparseTblis) {

   int dim = 16;

   Tensor<float> A("A", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> B("B", {dim, dim, dim}, Format{Sparse, Dense, Dense}); 
   Tensor<float> C("C", {dim, dim}, Format{Dense, Dense});

   float SPARSITY = .3;

   for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (float) ((int) (rand_float*3/SPARSITY)));
    }
   }


  for (int i = 0; i < dim; i++) {
   for (int j = 0; j < dim; j++) {
    for (int k = 0; k < dim; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, j, k}, (float) ((int) (rand_float*3/SPARSITY)));
      }
    }
   }
  }


   B.pack();
   C.pack();

   Tensor<float> expected("expected", {dim, dim, dim}, Format{Dense, Dense, Dense});

   TensorVar precomputed("precomputed", Type(taco::Float32, {Dimension(dim), Dimension(dim), Dimension(dim)}), Format{Dense, Dense, Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
   IndexVar l("l");
   IndexVar m("m");

   IndexExpr accelerateExpr = B(i, j, l) * C(l, k);
   A(i, j, k) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.holdConstant(new TblisMultiply(), accelerateExpr, {i}, precomputed(i, j, k));
   
   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i, j, k) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);


}


TEST(interface, sdmmBlas){
  int dim = 16;
  int NUM_I = dim;
  int NUM_K = dim;
  int NUM_J = dim;

  float SPARSITY = .3;
  
  Tensor<float> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<float> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<float> D("D", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<float> A("A", {NUM_I, NUM_K}, {Dense, Dense}, 0);
  Tensor<float> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});

  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (float) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  IndexVar i("i"), j("j"), k("k");

  IndexExpr accelerateExpr = C(i,j) * D(j,k);

  A(i,k) =  B(i,k) * accelerateExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.accelerate(new MatrixMultiply(), accelerateExpr);

   A.compile(stmt);
   A.assemble();
   A.compute();


   expected(i,k) =  B(i,k) * (accelerateExpr);

   expected.compile();
   expected.assemble();
   expected.compute();

  ASSERT_TENSOR_EQ(expected, A);

}


TEST(interface, sdmmGsl){
  gsl_compile = true;
  int dim = 16;
  int NUM_I = dim;
  int NUM_K = dim;
  int NUM_J = dim;

  float SPARSITY = .3;
  
  Tensor<float> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<float> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<float> D("D", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<float> A("A", {NUM_I, NUM_K}, {Dense, Dense}, 0);
  Tensor<float> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});

  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (float) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  IndexVar i("i"), j("j"), k("k");

  IndexExpr accelerateExpr = C(i,j) * D(j,k);

  A(i,k) =  B(i,k) * accelerateExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.accelerate(new GSLMM(), accelerateExpr);

   A.compile(stmt);
   A.assemble();
   A.compute();


   expected(i,k) =  B(i,k) * (accelerateExpr);

   expected.compile();
   expected.assemble();
   expected.compute();

  ASSERT_TENSOR_EQ(expected, A);
  gsl_compile = false;

}

TEST(interface, sdmmTblis){

  int dim = 16;
  int NUM_I = dim;
  int NUM_K = dim;
  int NUM_J = dim;

  float SPARSITY = .3;
  
  Tensor<float> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<float> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<float> D("D", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<float> A("A", {NUM_I, NUM_K}, {Dense, Dense}, 0);
  Tensor<float> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});

  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (float) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  IndexVar i("i"), j("j"), k("k");

  IndexExpr accelerateExpr = C(i,j) * D(j,k);

  A(i,k) =  B(i,k) * accelerateExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.accelerate(new TblisMultiply(), accelerateExpr);

   A.compile(stmt);
   A.assemble();
   A.compute();


   expected(i,k) =  B(i,k) * (accelerateExpr);

   expected.compile();
   expected.assemble();
   expected.compute();

  ASSERT_TENSOR_EQ(expected, A);


}

TEST(interface, tiledSaxpyAVX) {

   gsl_compile = false;

   Tensor<float> A("A", {16}, Format{Dense});
   Tensor<float> expected("expected", {16}, Format{Dense});
   Tensor<float> B("B", {16}, Format{Dense});
   Tensor<float> C("C", {16}, Format{Dense});
   IndexVar i("i");

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

   C.pack();
   B.pack();

   IndexExpr accelerateExpr = B(i) + C(i);
   A(i) = accelerateExpr;
   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.tile(new AVXSaxpy(), accelerateExpr, {{i,  8}});

   cout << stmt << endl;

   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);
}


TEST(interface, sdmmCblasDot){

  int dim = 16;
  int NUM_I = dim;
  int NUM_K = dim;
  int NUM_J = dim;

  float SPARSITY = .3;
  
  Tensor<float> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<float> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<float> D("D", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<float> A("A", {NUM_I, NUM_K}, {Dense, Dense}, 0);
  Tensor<float> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  TensorVar precomputed("precomputed", Type(taco::Float32, {16, 16}), Format{Dense, Dense});

  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (float) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  IndexVar i("i"), j("j"), k("k");

  IndexExpr accelerateExpr = C(i,j) * D(j,k);

   A(i,k) =  B(i,k) * accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.holdConstant(new Sdot(), accelerateExpr, {i, k}, precomputed(i, k));

   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i,k) =  B(i,k) * (accelerateExpr);

   expected.compile();
   expected.assemble();
   expected.compute();

  ASSERT_TENSOR_EQ(expected, A);

}

TEST(interface, sdmmTblisDot){

  int dim = 16;
  int NUM_I = dim;
  int NUM_K = dim;
  int NUM_J = dim;

  float SPARSITY = .3;
  
  Tensor<float> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<float> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<float> D("D", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<float> A("A", {NUM_I, NUM_K}, {Dense, Dense}, 0);
  Tensor<float> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  TensorVar precomputed("precomputed", Type(taco::Float32, {16, 16}), Format{Dense, Dense});

  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (float) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  IndexVar i("i"), j("j"), k("k");

  IndexExpr accelerateExpr = C(i,j) * D(j,k);

   A(i,k) =  B(i,k) * accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.holdConstant(new TblisDot(), accelerateExpr, {i, k}, precomputed(i, k));

   A.compile(stmt);
   A.assemble();
   A.compute();


   expected(i,k) =  B(i,k) * (accelerateExpr);

   expected.compile();
   expected.assemble();
   expected.compute();

  ASSERT_TENSOR_EQ(expected, A);

}

// TEST(interface, TTTPtest){

//   int dim = 16;
//   int NUM_I = dim;
//   int NUM_K = dim;
//   int NUM_J = dim;

//   float SPARSITY = .3;


//   Tensor<float> B("B", {NUM_I, NUM_K, NUM_K}, CSR)
//   Tensor<float> B("B", {NUM_I, NUM_K, NUM_K}, CSR);
//   Tensor<float> C1("C1", {NUM_I, NUM_J}, {Dense, Dense});
//   Tensor<float> C2("C2", {NUM_J, NUM_K}, {Dense, Dense});
//   Tensor<float> C3("A", {NUM_I, NUM_K}, {Dense, Dense}, 0);
//   Tensor<float> expected("expected", {NUM_I, NUM_K, NUM_K}, {Dense, Dense});
//   TensorVar precomputed("precomputed", Type(taco::Float32, {16, 16}), Format{Dense, Dense});


//   }

 
//   ASSERT_TENSOR_EQ(expected, A);

// }

TEST(interface, sdmmCblasGemv){

  int dim = 16;
  int NUM_I = dim;
  int NUM_K = dim;
  int NUM_J = dim;

  float SPARSITY = .3;
  
  Tensor<float> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<float> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<float> D("D", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<float> A("A", {NUM_I, NUM_K}, {Dense, Dense}, 0);
  Tensor<float> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  TensorVar precomputed("precomputed", Type(taco::Float32, {16, 16}), Format{Dense, Dense});

  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (float) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  IndexVar i("i"), j("j"), k("k");

  IndexExpr accelerateExpr = C(i,j) * D(j,k);

   A(i,k) =  B(i,k) * accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.holdConstant(new CblasGemv(), accelerateExpr, {k}, precomputed(i, k));

   A.compile(stmt);
   A.assemble();
   A.compute();


   expected(i,k) =  B(i,k) * (accelerateExpr);

   expected.compile();
   expected.assemble();
   expected.compute();

  ASSERT_TENSOR_EQ(expected, A);


}

TEST(interface, DimReduceBlockedSparse) {

   int dim = 4;

   Tensor<float> A("A", {dim, dim, dim, dim}, Format{Dense, Dense, Dense, Dense});
   Tensor<float> B("B", {dim, dim, dim, dim}, Format{Dense, Dense, Dense, Dense});
   Tensor<float> C("C", {dim, dim, dim, dim}, Format{Sparse, Dense, Sparse, Dense});

   TensorVar precomputed("precomputed", Type(taco::Float32, {4, 4, 4, 4}), Format{Dense, Dense, Dense, Dense});
   float SPARSITY = .4;
   for (int i = 0; i < dim; i++) {
    for (int k = 0; k < dim; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
         if (rand_float < SPARSITY) {
             for (int m = 0; m < dim; m++) {
                 for (int n = 0; n < dim; n++) {
                     B.insert({i, m, k, n}, (float) (float) 100);
                     C.insert({i, m, k, n}, (float) (float) 100);
                 }
            }
         }
    }
  }

   C.pack();
   B.pack();

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
   IndexVar l("l");
   IndexVar m("m");
   IndexVar n("n");

   IndexExpr accelerateExpr = B(i, k, j, l) * C(j, l, m, n);
   A(i, k, m, n) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.holdConstant(new MatrixMultiply(), accelerateExpr, {j, m, i}, precomputed(i, k, m, n));
   
   A.compile(stmt);
   A.assemble();
   A.compute();

}

TEST(interface, blasGmev) {

   // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> b("b", {16}, Format{Dense});
   Tensor<float> d("d", {16}, Format{Dense});

   for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
         A.insert({i, j}, (float) i + j);
      }
   }

   A.pack();

   for (int i = 0; i < 16; i++) {
      b.insert({i}, (float) i);
   }

   b.pack();

   Tensor<float> expected("expected", {16}, Format{Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = A(i, j) * b(j);
   d(i) = accelerateExpr;

   IndexStmt stmt = d.getAssignment().concretize();
   stmt = stmt.accelerate(new CblasGemv(), accelerateExpr, true);
   
   d.compile(stmt);
   d.assemble();
   d.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, d);

}

TEST(interface, gslGmev) {

   gsl_compile =true;

   // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> b("b", {16}, Format{Dense});
   Tensor<float> d("d", {16}, Format{Dense});

   for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
         A.insert({i, j}, (float) i + j);
      }
   }

   A.pack();

   for (int i = 0; i < 16; i++) {
      b.insert({i}, (float) i);
   }

   b.pack();

   Tensor<float> expected("expected", {16}, Format{Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = A(i, j) * b(j);
   d(i) = accelerateExpr;

   IndexStmt stmt = d.getAssignment().concretize();
   stmt = stmt.accelerate(new GSLGemv(), accelerateExpr, true);
   
   d.compile(stmt);
   d.assemble();
   d.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, d);
   gsl_compile = false;
   
}

TEST(interface, tensorPlusSampleCode) {

   gsl_compile = true;

   int dim = 16;

   // actual computation
   Tensor<float> A("A", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> B("B", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> C("C", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> D("D", {dim, dim, dim}, Format{Dense, Dense, Dense});
   Tensor<float> expected("expected", {dim, dim, dim}, Format{Dense, Dense, Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
   IndexVar l("l");

   for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
         for (int k = 0; k < dim; k++) {
            B.insert({i, j, k}, (float) i + j * k);
            C.insert({i, j, k}, (float) i + j * k);
         }
      }
   }

   C.pack();
   B.pack();

   IndexExpr accelerateExpr = B(i, j, k) + C(i, j, k);
   A(i, j, k) = accelerateExpr + D(i, j, k);


   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new TblisPlus(), accelerateExpr);
   
   A.compile(stmt);
   A.assemble();
   A.compute();

   // expected(i, j, k) = accelerateExpr;
   // expected.compile();
   // expected.assemble();
   // expected.compute();

   // ASSERT_TENSOR_EQ(expected, A);

   gsl_compile = false;

}


TEST(interface, symmtericBlasGemv) {

   // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> b("b", {16}, Format{Dense});
   Tensor<float> d("d", {16}, Format{Dense});

   for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
         A.insert({i, j}, (float) i + j);
      }
   }

   A.pack();

   for (int i = 0; i < 16; i++) {
      b.insert({i}, (float) i);
   }

   b.pack();

   Tensor<float> expected("expected", {16}, Format{Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = A(i, j) * b(j);
   d(i) = accelerateExpr;

   IndexStmt stmt = d.getAssignment().concretize();
   stmt = stmt.accelerate(new CblasSymmetricGemV(), accelerateExpr, true);
   
   d.compile(stmt);
   d.assemble();
   d.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, d);

}



TEST(interface, symmtericGSLGemv) {

   // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> b("b", {16}, Format{Dense});
   Tensor<float> d("d", {16}, Format{Dense});

   for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
         A.insert({i, j}, (float) i + j);
      }
   }

   A.pack();

   for (int i = 0; i < 16; i++) {
      b.insert({i}, (float) i);
   }

   b.pack();

   Tensor<float> expected("expected", {16}, Format{Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = A(i, j) * b(j);
   d(i) = accelerateExpr;

   IndexStmt stmt = d.getAssignment().concretize();
   stmt = stmt.accelerate(new GSLSymmetricGemv(), accelerateExpr, true);
   
   d.compile(stmt);
   d.assemble();
   d.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, d);

}

TEST(interface, DimReduceBlockedSparseDense) {

   int dim = 4;

   Tensor<float> A("A", {dim, dim, dim, dim}, Format{Dense, Dense, Dense, Dense});
   Tensor<float> B("B", {dim, dim, dim, dim}, Format{Sparse, Dense, Sparse, Dense});
   Tensor<float> C("C", {dim, dim, dim, dim}, Format{Dense, Dense, Dense, Dense});

   TensorVar precomputed("precomputed", Type(taco::Float32, {4, 4, 4, 4}), Format{Dense, Dense, Dense, Dense});
   float SPARSITY = .4;
   for (int i = 0; i < dim; i++) {
    for (int k = 0; k < dim; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
         if (rand_float < SPARSITY) {
             for (int m = 0; m < dim; m++) {
                 for (int n = 0; n < dim; n++) {
                     B.insert({i, m, k, n}, (float) (float) 100);
                     C.insert({i, m, k, n}, (float) (float) 100);
                 }
            }
         }
    }
  }

   C.pack();
   B.pack();

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
   IndexVar l("l");
   IndexVar m("m");
   IndexVar n("n");

   IndexExpr accelerateExpr = B(i, k, j, l) * C(j, l, m, n);
   A(i, k, m, n) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.holdConstant(new MatrixMultiply(), accelerateExpr, {j, m, i}, precomputed(i, k, m, n));
   
   A.compile(stmt);
   A.assemble();
   A.compute();

}

TEST(interface, tblisSgmev) {
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

   IndexExpr accelerateExpr = A(i, j) * b(j);
   d(i) = accelerateExpr;


   IndexStmt stmt = d.getAssignment().concretize();
   stmt = stmt.accelerate(new TblisGemv(), accelerateExpr, true);
   
   d.compile(stmt);
   d.assemble();
   d.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, d);

}

TEST(interface, mklGemv) {
   mkl_compile = true;
   cout << "MKL COMPILE 2!!" << mkl_compile << endl;
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

   IndexExpr accelerateExpr = A(i, j) * b(j);
   d(i) = accelerateExpr;


   IndexStmt stmt = d.getAssignment().concretize();
   stmt = stmt.accelerate(new MklSgemv(), accelerateExpr, true);
   cout << "running";
   d.compile(stmt);
   cout << "running";
   d.assemble();
   cout << "running";
   d.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, d);
   mkl_compile = false;

}

TEST(interface, mklSparseGemv) {
   mkl_compile = true;
   // actual computation
   Tensor<float> A("A", {16, 16}, CSR);
   Tensor<float> b("b", {16}, Format{Dense});
   Tensor<float> d("d", {16}, Format{Dense});

   for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
         A.insert({i, j}, (float) i + j);
      }
   }

   A.pack();

   for (int i = 0; i < 16; i++) {
      b.insert({i}, (float) i);
   }

   b.pack();

   Tensor<float> expected("expected", {16}, Format{Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = A(i, j) * b(j);
   d(i) = accelerateExpr;


   IndexStmt stmt = d.getAssignment().concretize();
   stmt = stmt.accelerate(new SparseMklSgemv(), accelerateExpr, true);
   d.compile(stmt);
   d.assemble();
   d.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, d);
   mkl_compile = false;

}


TEST(interface, sdmmMKL){
  int dim = 16;
  int NUM_I = dim;
  int NUM_K = dim;
  int NUM_J = dim;

  float SPARSITY = .3;
  
  Tensor<float> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<float> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<float> D("D", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<float> A("A", {NUM_I, NUM_K}, {Dense, Dense}, 0);
  Tensor<float> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});

  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (float) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  IndexVar i("i"), j("j"), k("k");

  IndexExpr accelerateExpr = B(i,j) * C(j,k);

  A(i,k) = accelerateExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.accelerate(new SparseMklMM(), accelerateExpr);

   A.compile(stmt);
   A.assemble();
   auto func = A.compute_split();
   auto pair = A.returnFuncPackedRaw(func);
   pair.first(func.data());


   expected(i,k) = (accelerateExpr);

   expected.compile();
   expected.assemble();
   expected.compute();

  ASSERT_TENSOR_EQ(expected, A);

}

TEST(interface, cudaSparseGemv) {
   mkl_compile = true;
   // actual computation
   Tensor<float> A("A", {16, 16}, CSR);
   Tensor<float> b("b", {16}, Format{Dense});
   Tensor<float> d("d", {16}, Format{Dense});

   for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
         A.insert({i, j}, (float) i + j);
      }
   }

   A.pack();

   for (int i = 0; i < 16; i++) {
      b.insert({i}, (float) i);
   }

   b.pack();

   Tensor<float> expected("expected", {16}, Format{Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = A(i, j) * b(j);
   d(i) = accelerateExpr;


   IndexStmt stmt = d.getAssignment().concretize();
   stmt = stmt.accelerate(new CudaSpmv(), accelerateExpr, true);
   d.compile(stmt);
   d.assemble();
   d.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, d);
   mkl_compile = false;

}

TEST(interface, symmtericMklGemv) {

   // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> b("b", {16}, Format{Dense});
   Tensor<float> d("d", {16}, Format{Dense});

   for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
         A.insert({i, j}, (float) i + j);
      }
   }

   A.pack();

   for (int i = 0; i < 16; i++) {
      b.insert({i}, (float) i);
   }

   b.pack();

   Tensor<float> expected("expected", {16}, Format{Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = A(i, j) * b(j);
   d(i) = accelerateExpr;

   IndexStmt stmt = d.getAssignment().concretize();
   stmt = stmt.accelerate(new MklSymmgemv(), accelerateExpr, true);
   
   d.compile(stmt);
   d.assemble();
   d.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, d);

}

TEST(interface, sdmmMkl){
  int dim = 16;
  int NUM_I = dim;
  int NUM_K = dim;
  int NUM_J = dim;

  float SPARSITY = .3;
  
  Tensor<float> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<float> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<float> D("D", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<float> A("A", {NUM_I, NUM_K}, {Dense, Dense}, 0);
  Tensor<float> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});

  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      C.insert({i, j}, (float) i+j);
      D.insert({i, j}, (float) i+j);
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (float) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  B.pack();
  C.pack();
  D.pack();

  IndexVar i("i"), j("j"), k("k");

  IndexExpr accelerateExpr = C(i,j) * D(j,k);

  A(i,k) =  B(i,k) * accelerateExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.accelerate(new MklMM(), accelerateExpr);

   A.compile(stmt);
   A.assemble();
   A.compute();


   expected(i,k) =  B(i,k) * accelerateExpr;

   expected.compile();
   expected.assemble();
   expected.compute();

  ASSERT_TENSOR_EQ(expected, A);

}


TEST(interface, mklDot) {


   Tensor<float> A("A", {16}, Format{Dense}, 0);
   Tensor<float> B("B", {16}, Format{Dense});
   Tensor<float> C("C", {16}, Format{Dense});
   Tensor<float> expected("expected", {16}, Format{Dense}, 0);
   TensorVar accelWorkspace("accelWorkspace", Type(taco::Float32, {16}), taco::dense);
   float SPARSITY = .3;

   IndexVar i("i");
   IndexVar j("j");
   IndexVar iw("iw");

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

   C.pack();
   B.pack();

   IndexExpr accelerateExpr = B(j) * C(j);
   A(i) = sum(j, accelerateExpr);

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new MklDot(), accelerateExpr);
    
   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i) = sum(j, accelerateExpr);
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);
}



TEST(interface, MMAddSparse) {

   int dim = 16;

   Tensor<float> A("A", {dim, dim}, CSR);
   Tensor<float> B("B", {dim, dim}, CSR);
   Tensor<float> C("C", {dim, dim}, CSR);
   IndexVar i("i");
   IndexVar j("j");
  

  float SPARSITY = 0.3;
  for (int i = 0; i < dim; i++) {
    for (int k = 0; k < dim; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (float) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

   IndexExpr expr = B(i, j) + C(i, j);
   A(i, j) = expr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new MklAdd(), expr, true);

   A.compile(stmt);
   A.assemble();
   A.compute();
    

}


TEST(interface, TTVBlas) {

   int dim = 16;

   Tensor<float> A("A", {dim, dim}, {Dense, Dense});
   Tensor<float> B("B", {dim, dim, dim}, {Dense, Dense, Dense});
   Tensor<float> C("C", {dim}, {Dense});
   TensorVar precomputed("precomputed", Type(taco::Float32, {16, 16}), Format{Dense, Dense});
   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
  

//   float SPARSITY = 0.3;
//   for (int i = 0; i < dim; i++) {
//     for (int k = 0; k < dim; k++) {
//       float rand_float = (float)rand()/(float)(RAND_MAX);
//       if (rand_float < SPARSITY) {
//         B.insert({i, k}, (float) ((int) (rand_float*3/SPARSITY)));
//       }
//     }
//   }

   IndexExpr expr = B(i, j, k) * C(k);
   A(i, j) = expr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.holdConstant(new CblasGemv(), expr, {i}, precomputed(i, j));

   A.compile(stmt);
   A.assemble();
   A.compute();
    

}

TEST(interface, TTVTblis) {

   int dim = 16;

   Tensor<float> A("A", {dim, dim}, {Dense, Dense});
   Tensor<float> B("B", {dim, dim, dim}, {Dense, Dense, Dense});
   Tensor<float> C("C", {dim}, {Dense});
   TensorVar precomputed("precomputed", Type(taco::Float32, {16, 16}), Format{Dense, Dense});
   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
  

//   float SPARSITY = 0.3;
//   for (int i = 0; i < dim; i++) {
//     for (int k = 0; k < dim; k++) {
//       float rand_float = (float)rand()/(float)(RAND_MAX);
//       if (rand_float < SPARSITY) {
//         B.insert({i, k}, (float) ((int) (rand_float*3/SPARSITY)));
//       }
//     }
//   }

   IndexExpr expr = B(i, j, k) * C(k);
   A(i, j) = expr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new TblisTTV(), expr);

   A.compile(stmt);
   A.assemble();
   A.compute();
    

}


TEST(interface, sdmmMKL_COO_to_CSR){
  int dim = 300;
  int NUM_I = dim;
  int NUM_K = dim;
  int NUM_J = dim;

  float SPARSITY = .05;
  
  Tensor<float> B("B", {NUM_I, NUM_K}, COO(2));
  Tensor<float> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<float> D("D", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<float> A("A", {NUM_I, NUM_K}, {Dense, Dense}, 0);
  Tensor<float> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});

  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  std::mt19937 mt(0); 
  
//   float SPARSITY = .5;
   for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      const float randnum = mt();
      float rand_float = randnum/(float)(mt.max());
         if (rand_float < SPARSITY) {
                     B.insert({i, k}, (float) (float) 100);
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  IndexVar i("i"), j("j"), k("k");

  IndexExpr accelerateExpr = B(i,j) * C(j,k);

  A(i,k) = accelerateExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.accelerate(new SparseMklMMCOOCSR(), accelerateExpr);

   A.compile(stmt);
   A.assemble();
   auto func = A.compute_split();
   auto pair = A.returnFuncPackedRaw(func);
   pair.first(func.data());

   expected(i,k) = (accelerateExpr);

   expected.compile();
   expected.assemble();
   expected.compute();

  ASSERT_TENSOR_EQ(expected, A);

}