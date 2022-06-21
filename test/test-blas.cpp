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

   Tensor<uint32_t> test_a("actuala", {3}, dense, 0);
   Tensor<uint32_t> test_b("actualb", {3}, dense, 1);
   Tensor<uint32_t> test_c("actualc", {3}, dense, 1);

   TensorVar workspace(Type(taco::UInt32, {1}), taco::dense, 0) ;

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

   auto it = iterate<uint32_t>(test_c);
   auto iit = it.begin();

   while (iit != it.end()){
      cout << "val " << iit->second << endl;
      ++iit;
   }

   // test_c.evaluateAccelerated(canAccelerate);

   // cout << test_c << endl;

//    test_c.evaluateAccelerated(canAccelerate);

}

