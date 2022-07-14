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
#include "taco/lower/lower.h"


using namespace taco;


TEST(accelerateSchedule, targetAccelerate) {

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
//   stmt = stmt.accelerate(accelerateExpr, i, iw, accelWorkspace);

   cout << stmt << endl;

}


TEST(completeProcess, targetAccelerate) {

   TensorVar a("a", Type(taco::Float32, {Dimension()}), taco::dense);
   TensorVar b("a", Type(taco::Float32, {Dimension()}), taco::dense);
   IndexVar i("i");

   Tensor<float32_t> test_a("actuala", {3}, dense, 0);
   Tensor<float32_t> test_b("actualb", {3}, dense, 1);
   Tensor<float32_t> test_c("actualc", {3}, dense, 1);


   IndexStmt stmt = test_c(i) = test_a(i) + test_b(i);

   std::vector<IndexExpr> canAccelerate = {a(i) + b(i)};

   test_c.compileAccelerated(canAccelerate);
   test_c.assemble();
   test_c.compute();

}





