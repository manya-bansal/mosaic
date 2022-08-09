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
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/lower/lower.h"
#include "taco/ir_tags.h"


using namespace taco;



TEST(blasTest, simpleBlasCall) {

 

   TensorVar a("a", Type(taco::Float32, {Dimension()}), taco::dense);
   TensorVar b("a", Type(taco::Float32, {Dimension()}), taco::dense);
   IndexVar i("i");

   Tensor<float32_t> test_a("actuala", {3}, dense, 0);
   Tensor<float32_t> test_b("actualb", {3}, dense, 1);
   Tensor<float32_t> test_c("actualc", {3}, dense, 1);

   TensorVar workspace(Type(taco::Float32, {1}), taco::dense, 0) ;

   IndexStmt stmt = test_c(i) = test_a(i) + test_b(i);
   // IndexStmt stmt_accel = test_c(i) = test_a(i)  test_b(i);

   std::vector<IndexExpr> canAccelerate = {a(i) + b(i)};


   // stmt = stmt.concretize();
   // stmt = stmt.precompute(test_a(i) + test_b(i), i, IndexVar(), workspace);
   // cout << stmt << endl;

   // makeConcreteNotation(stmt);
   // makeAcceleratedConcreteNotation(stmt_accel, canAccelerate);

   // std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);

   test_c.compileAccelerated(canAccelerate);
   test_c.assemble();
   test_c.compute();

   auto it = iterate<float32_t>(test_c);
   auto iit = it.begin();

   while (iit != it.end()){
      cout << "val " << iit->second << endl;
      ++iit;
   }
}

template <typename T>
ir::Expr makeTensorArg(T t){
      return ir::Var::make(t.getName(), t.getComponentType(),true, true);
}

ir::Expr makeTensorArgVar(TensorVar t){
   return ir::Var::make(t.getName(), t.getType().getDataType(),true, true);
}




bool trivialChecker(IndexExpr expr){
   return true;
}

TEST(accelerateScheduleLower, simpleBlasCallFunction) {

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
  TensorVar accelWorkspace("accelWorkspace", Type(taco::Float32, {16}), taco::dense);

  std::function<bool(IndexExpr)> checker = trivialChecker;

  std::vector<taco::ir::Expr> args = {makeTensorArg(B), makeTensorArg(C), makeTensorArgVar(accelWorkspace)};

   ConcreteAccelerateCodeGenerator concrete_cblas_saxpy("cblas_saxpy", "void",  B(i),  B(i) + C(i), {});
//   ConcreteAccelerateCodeGenerator accelGen(accelerateExpr, "add", args, checker);

   stmt = stmt.accelerate(concrete_cblas_saxpy, i, iw, accelWorkspace);

   cout << stmt << endl;

   // A.compile(stmt);
   // A.assemble();
   // A.compute();

   // auto it = iterate<float32_t>(A);
   // auto iit = it.begin();

   // while (iit != it.end()){
   //    cout << "val " << iit->second << endl;
   //    ++iit;
   // }

}



TEST(accelerateScheduleLower, testExpr) {

   int NUM_I = 100;
  int NUM_J = 100;
  int NUM_K = 100;

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
   IndexVar l("l");

  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<double> B("B", {NUM_J, 100}, {Dense, Dense});
  Tensor<double> D("D", {NUM_I, NUM_K}, {Dense, Dense});

  expected(i, k) = A(i, j) * B(j, k);

//   stmt = stmt.concretize();



//   stmt = makeConcreteNotation(stmt);

   // Tensor<double> test("test", {NUM_I, NUM_J, NUM_K, NUM_K}, {Dense, Dense, Dense, Dense});

   // test(i, j, k, l) = A(i, l) * B(l ,k);

//   cout << stmt << endl;

  expected.compile();
  expected.assemble();
  expected.compute();
//   ASSERT_TENSOR_EQ(expected, C);
}

TEST(accelerateScheduleLower, testExpr2) {

   int NUM_I = 100;
  int NUM_J = 100;
  int NUM_K = 100;

   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");
   IndexVar l("l");

  Tensor<double> expected("expected", {NUM_I, NUM_J, NUM_K}, {Dense, Dense, Dense});
  Tensor<double> A("A", {NUM_I, NUM_J, NUM_J}, {Dense, Dense, Dense});
  Tensor<double> B("B", {NUM_J, NUM_K, NUM_J}, {Dense, Dense, Dense});
//   Tensor<double> D("D", {NUM_I, NUM_K}, {Dense, Dense});

  expected(i, j, k) = A(l, j, k) * B(i, l, k);

//   stmt = stmt.concretize(x);



//   stmt = makeConcreteNotation(stmt);

   // Tensor<double> test("test", {NUM_I, NUM_J, NUM_K, NUM_K}, {Dense, Dense, Dense, Dense});

   // test(i, j, k, l) = A(i, l) * B(l ,k);

//   cout << stmt << endl;

  expected.compile();
  expected.assemble();
  expected.compute();
//   ASSERT_TENSOR_EQ(expected, C);
}

