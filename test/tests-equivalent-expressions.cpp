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
#include "taco/lower/lower.h"

using namespace taco;

TEST(generateEquivExpressions, addExpr) {

   TensorVar a("a", Type(taco::Float32, {Dimension()}), taco::dense);
   TensorVar b("a", Type(taco::Float32, {Dimension()}), taco::dense);
   IndexVar i("i");

   Tensor<float> actuala("actuala", {3}, dense, 0);
   Tensor<float> actualb("actualb", {3}, dense, 1);
   Tensor<float> actualc("actualc", {3}, dense, 1);


   IndexStmt stmt = actualc(i) = actuala(i) + actualb(i) + actualb(i);

   std::cout << util::join(generateEquivalentStmts(stmt)) << std::endl;

   stmt = actualc(i) = (actuala(i) + actualb(i));

  std::cout << util::join(generateEquivalentStmts(stmt)) << std::endl;

}

TEST(generateEquivExpressions, subExpr) {

   TensorVar a("a", Type(taco::Float32, {Dimension()}), taco::dense);
   TensorVar b("a", Type(taco::Float32, {Dimension()}), taco::dense);
   IndexVar i("i");

   Tensor<float> actuala("actuala", {3}, dense, 0);
   Tensor<float> actualb("actualb", {3}, dense, 1);
   Tensor<float> actualc("actualc", {3}, dense, 1);


   IndexStmt stmt = actualc(i) = actuala(i) - actualb(i);

  std::cout << util::join(generateEquivalentStmts(stmt)) << std::endl;

}


TEST(generateEquivExpressions, addMulExpr) {

   TensorVar a("a", Type(taco::Float32, {Dimension()}), taco::dense);
   TensorVar b("a", Type(taco::Float32, {Dimension()}), taco::dense);
   IndexVar i("i");

   Tensor<float> actuala("actuala", {3}, dense, 0);
   Tensor<float> actualb("actualb", {3}, dense, 1);
   Tensor<float> actualc("actualc", {3}, dense, 1);


   IndexStmt stmt = actualc(i) = actuala(i) + actualb(i) * 5;

  std::cout << util::join(generateEquivalentStmts(stmt)) << std::endl;

   stmt = actualc(i) = (actuala(i) + actualb(i)) * 5;

  std::cout << util::join(generateEquivalentStmts(stmt)) << std::endl;

}


TEST(generateEquivExpressions, mulExpr) {

   TensorVar a("a", Type(taco::Float32, {Dimension()}), taco::dense);
   TensorVar b("a", Type(taco::Float32, {Dimension()}), taco::dense);
   IndexVar i("i");

   Tensor<float> actuala("actuala", {3}, dense, 0);
   Tensor<float> actualb("actualb", {3}, dense, 1);
   Tensor<float> actualc("actualc", {3}, dense, 1);


   IndexStmt stmt = actualc(i) = actuala(i) * actualb(i) * 5;

  std::cout << util::join(generateEquivalentStmts(stmt)) << std::endl;

}



TEST(generateEquivExpressions, divExpr) {

   TensorVar a("a", Type(taco::Float32, {Dimension()}), taco::dense);
   TensorVar b("a", Type(taco::Float32, {Dimension()}), taco::dense);
   IndexVar i("i");

   Tensor<float> actuala("actuala", {3}, dense, 0);
   Tensor<float> actualb("actualb", {3}, dense, 1);
   Tensor<float> actualc("actualc", {3}, dense, 1);


   IndexStmt stmt = actualc(i) = (actuala(i) + actualb(i)) / 5;

  std::cout << util::join(generateEquivalentStmts(stmt)) << std::endl;

   stmt = actualc(i) = (actuala(i) * actualb(i)) / 5;

   std::cout << util::join(generateEquivalentStmts(stmt)) << std::endl;

}

TEST(generateEquivExpressions, takeCommonTerms) {

   TensorVar a("a", Type(taco::Float32, {Dimension()}), taco::dense);
   TensorVar b("a", Type(taco::Float32, {Dimension()}), taco::dense);
   IndexVar i("i");

   Tensor<float> actuala("actuala", {3}, dense, 0);
   Tensor<float> actualb("actualb", {3}, dense, 1);
   Tensor<float> actualc("actualc", {3}, dense, 1);


   IndexStmt stmt = actualc(i) = (actuala(i)*5 + actualb(i)*5);
  std::cout << util::join(generateEquivalentStmts(stmt)) << std::endl;

    stmt = actualc(i) = (actuala(i)/5 + actualb(i)/5);
   std::cout << util::join(generateEquivalentStmts(stmt)) << std::endl;


    stmt = actualc(i) = (actuala(i)/5 - actualb(i)/5);
   std::cout << util::join(generateEquivalentStmts(stmt)) << std::endl;

    stmt = actualc(i) = (actuala(i)*5 - actualb(i)*5);
   std::cout << util::join(generateEquivalentStmts(stmt)) << std::endl;


}

// TEST(generateEquivExpressions, simplifyNegatives) {

//    TensorVar a("a", Type(taco::Float32, {Dimension()}), taco::dense);
//    TensorVar b("a", Type(taco::Float32, {Dimension()}), taco::dense);
//    IndexVar i("i");

//    Tensor<float> actuala("actuala", {3}, dense, 0);
//    Tensor<float> actualb("actualb", {3}, dense, 1);
//    Tensor<float> actualc("actualc", {3}, dense, 1);


//    IndexStmt stmt = actualc(i) = (actuala(i) + (-actualb(i));
//   std::cout << util::join(generateEquivalentStmts(stmt)) << std::endl;

// }