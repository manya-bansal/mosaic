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
#include "taco/index_notation/accel_interface.h"
#include "codegen/codegen.h"
#include "taco/lower/lowerer_impl_imperative.h"
#include "taco/index_notation/accelerate_notation.h"
#include "taco/lower/lower.h"
#include "taco/ir_tags.h"


using namespace taco;


bool trivialkernelChecker(IndexStmt expr){
   return true;
}


TEST(transferType, pluginInterface) {

    TensorVar a("a", Type(taco::Float32, {Dimension()}), taco::dense);
    TensorVar b("b", Type(taco::Float32, {Dimension()}), taco::dense);
    IndexVar i("i");

    // should basically call a C function 
    // that can be included in header
    TransferLoad load_test("load_test", "void");
    TransferStore store_test("store_test", "void");

    TransferType kernelTransfer("test", load_test, store_test);

    ForeignFunctionDescription kernel1(a(i) = a(i) + b(i), "kernel1", "void", {}, trivialkernelChecker);
    ForeignFunctionDescription kernel2(a(i) = b(i), "kernel2", "void", {}, trivialkernelChecker);

   //  AcceleratorDescription accelDesc(kernelTransfer, 
   //          {  kernel1(load_test(a)),
   //             kernel2(load_test(a, load_test(b)), load_test(new TensorPropertiesArgs(b)))
   //          });

   cout << load_test(a, load_test(a), load_test(a), a) << endl;

    //need to register AcceleratorDescription
    //so that the TACO can use it

}