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


TEST(time, timeEndToEndSaxpy) {

   int NUM_I = 10;
   int NUM_K = 10;
   int NUM_J = 10;

   Tensor<float> A("A", {NUM_K}, {taco::Dense});
   Tensor<float> B("B", {NUM_K}, {taco::Dense});
   Tensor<float> C("C", {NUM_K}, {taco::Dense});

   IndexVar i("i"), j("j"), k("k"), l("l");

   A(i) = B(i) + C(i);


   // register the description
   A.registerAccelerator(new Saxpy());
   A.registerAccelerator(new Sdot());
   A.registerAccelerator(new Sgemm());
   // enable targeting
   A.accelerateOn();

   A.compile();
   A.assemble();

}

TEST(time, timeEndToEndSaxpyLiteral) {

   int NUM_I = 10;
   int NUM_K = 10;
   int NUM_J = 10;

   Tensor<float> A("A", {NUM_K}, {taco::Dense});
   Tensor<float> B("B", {NUM_K}, {taco::Dense});
   Tensor<float> C("C", {NUM_K}, {taco::Dense});

   IndexVar i("i"), j("j"), k("k"), l("l");

   A(i) = B(i) + 5 * C(i);


   // register the description
   A.registerAccelerator(new Saxpy());
   A.registerAccelerator(new Sdot());
   A.registerAccelerator(new Sgemm());
   // enable targeting
   A.accelerateOn();

   A.compile();
   A.assemble();

}

TEST(time, timeEndToEndDot) {

   int NUM_I = 10;
   int NUM_K = 10;
   int NUM_J = 10;

   Tensor<float> A("A");
   Tensor<float> B("B", {NUM_K}, {taco::Dense});
   Tensor<float> C("C", {NUM_K}, {taco::Dense});

   IndexVar i("i"), j("j"), k("k"), l("l");

   A = B(i) * C(i);


   // register the description
   A.registerAccelerator(new Saxpy());
   A.registerAccelerator(new Sdot());
   A.registerAccelerator(new Sgemm());
   // enable targeting
   A.accelerateOn();

   A.compile();
   A.assemble();

}

TEST(time, timeEndToEndGemv) {

   int NUM_I = 10;
   int NUM_K = 10;
   int NUM_J = 10;

   Tensor<float> A("A", {NUM_K}, {taco::Dense});
   Tensor<float> B("B", {NUM_K, NUM_K}, {taco::Dense, taco::Dense});
   Tensor<float> C("C", {NUM_K}, {taco::Dense});

   IndexVar i("i"), j("j"), k("k"), l("l");

   A(i) = B(i,j) * C(j);


   // register the description
   A.registerAccelerator(new Saxpy());
   A.registerAccelerator(new Sdot());
   A.registerAccelerator(new Sgemm());
   // enable targeting
   A.accelerateOn();

   A.compile();
   A.assemble();

}

TEST(time, timeEndToEndTTM) {

   int NUM_I = 10;
   int NUM_K = 10;
   int NUM_J = 10;

   Tensor<float> A("A", {NUM_K, NUM_K, NUM_K}, {taco::Dense, taco::Dense, taco::Dense});
   Tensor<float> B("B", {NUM_K, NUM_K, NUM_K}, {taco::Dense, taco::Dense, taco::Dense});
   Tensor<float> C("C", {NUM_K, NUM_K}, {taco::Dense, taco::Dense});

   IndexVar i("i"), j("j"), k("k"), l("l");

   A(i, j, k) = B(i,j, l) * C(k, l);


   // register the description
   A.registerAccelerator(new Saxpy());
   A.registerAccelerator(new Sdot());
   A.registerAccelerator(new Sgemm());
   // enable targeting
   A.accelerateOn();

   A.compile();
   A.assemble();

}

TEST(time, timeEndToEndPlus3) {

   int NUM_I = 10;
   int NUM_K = 10;
   int NUM_J = 10;

   Tensor<float> A("A", {NUM_K, NUM_K, NUM_K}, {taco::Dense, taco::Dense, taco::Dense});
   Tensor<float> B("B", {NUM_K, NUM_K, NUM_K}, {taco::Dense, taco::Dense, taco::Dense});
   Tensor<float> C("C", {NUM_K, NUM_K, NUM_K}, {taco::Dense, taco::Dense, taco::Dense});

   IndexVar i("i"), j("j"), k("k"), l("l");

   A(i, j, k) = B(i,j, k) +  C(i, j, k);


   // register the description
   A.registerAccelerator(new Saxpy());
   A.registerAccelerator(new Sdot());
   A.registerAccelerator(new Sgemm());
   // enable targeting
   A.accelerateOn();

   A.compile();
   A.assemble();

}


TEST(time, timeEndToEndMMScalar) {

   int NUM_I = 10;
   int NUM_K = 10;
   int NUM_J = 10;

   Tensor<float> A("A", {NUM_K, NUM_K}, {taco::Dense, taco::Dense});
   Tensor<float> B("B", {NUM_K, NUM_K}, {taco::Dense, taco::Dense});
   Tensor<float> C("C", {NUM_K, NUM_K}, {taco::Dense, taco::Dense});

   IndexVar i("i"), j("j"), k("k"), l("l");

   A(i,k) = 5 * B(i,j) * C(j, k) + 5 * C(i,k);


   // register the description
   A.registerAccelerator(new Saxpy());
   A.registerAccelerator(new Sdot());
   A.registerAccelerator(new Sgemm());
   // enable targeting
   A.accelerateOn();

   A.compile();
   A.assemble();

}