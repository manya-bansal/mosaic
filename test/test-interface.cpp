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
#include "taco/accelerator_interface/tile_interface.h"


using namespace taco;


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

   cout << load_test(a, load_test(a), load_test(a, load_test(a)), b, Dim(i)) << endl;

   Tensor<double> A("A", {16}, Format{Dense});

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

TEST(DISABLED_interface, tiledSaxpyInterface) {


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
   stmt = stmt.accelerate(new TileSaxpy(), accelerateExpr);

   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i) = accelerateExpr;
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

TEST(interface, cblasSgmev) {

   // actual computation
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> b("b", {16}, Format{Dense});
   Tensor<float> c("c", {16}, Format{Dense});
   Tensor<float> d("d", {16}, Format{Dense});

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
   Tensor<float> D("C", {16, 16}, Format{Dense, Dense});

   Tensor<float> expected("expected", {16, 16}, Format{Dense, Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");

   IndexExpr accelerateExpr = B(i, j) * C(j, k) + C(i,k);
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
   Tensor<float> A("A", {16, 16, 16}, Format{Dense, Dense, Dense});
   Tensor<float> B("B", {16, 16, 16}, Format{Dense, Dense, Dense});
   Tensor<float> C("C", {16, 16}, Format{Dense, Dense});
   Tensor<float> expected("expected", {16, 16, 16}, Format{Dense, Dense, Dense});

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
   IndexVar l("l");

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