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
#include "taco/index_notation/accelerate_notation.h"
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


TEST(functionObject, simpleBlasCall) {

 

   Tensor<float32_t> test_a("actuala", {3}, dense, 0);
   Tensor<float32_t> test_b("actualb", {3}, dense, 1);
   Tensor<float32_t> test_c("actualc", {3}, dense, 1);
   IndexVar i("i");

   TensorVar workspace(Type(taco::Float32, {1}), taco::dense, 0) ;

   IndexStmt stmt = test_c(i) = test_a(i) + test_b(i);


   // vector<Expr> args = {ir::Var::make(test_a.getName(), test_a.getComponentType(),true, true)};

   std::vector<taco::ir::Expr> args = {ir::Var::make(test_a.getName(), test_a.getComponentType(),true, true)};;

   // AccelerateCodeGenerator accelerateSpec()

}

bool trivialChecker(IndexExpr expr){
   return true;
}

TEST(accelerateScheduleLower, simpleBlasCall) {

  Tensor<double> A("A", {16}, Format{Dense});
  Tensor<double> B("B", {16}, Format{Dense});
  Tensor<double> C("C", {16}, Format{Dense});

  for (int i = 0; i < 16; i++) {
      A.insert({i}, (double) i);
      B.insert({i}, (double) i);
  }

  A.pack();
  B.pack();

  IndexVar i("i");
  IndexVar iw("iw");
  IndexExpr accelerateExpr = B(i) + C(i);
  A(i) = accelerateExpr;


  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar accelWorkspace("accelWorkspace", Type(taco::Float64, {16}), taco::dense);

  std::function<bool(IndexExpr)> checker = trivialChecker;

  std::vector<taco::ir::Expr> args = {makeTensorArg(A), makeTensorArg(B), makeTensorArgVar(accelWorkspace)};


  AccelerateCodeGenerator accelGen(accelerateExpr, "add", args, checker);

   stmt = stmt.accelerate(accelGen, i, iw, accelWorkspace);

   A.compile(stmt);
   A.assemble();
   A.compute();

}

