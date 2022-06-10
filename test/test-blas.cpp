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



TEST(blasTest, simpleBlasCall) {


   TensorVar a("a", Type(taco::UInt32, {Dimension()}), taco::dense);
   TensorVar b("a", Type(taco::UInt32, {Dimension()}), taco::dense);
   IndexVar i("i");

   Tensor<uint32_t> test_a("actuala", {1}, dense, 0);
   Tensor<uint32_t> test_b("actualb", {1}, dense, 1);
   Tensor<uint32_t> test_c("actualc", {1}, dense, 1);

   IndexStmt stmt = test_c(i) = test_a(i) + test_b(i);

   std::vector<IndexExpr> canAccelerate = {a(i) + b(i)};

   makeAcceleratedConcreteNotation(stmt, canAccelerate);

//    test_c.evaluateAccelerated(canAccelerate);




}

