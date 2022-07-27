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

ir::Expr makeTensorArgVarLocal(TensorVar t){
   return ir::Var::make(t.getName(), t.getType().getDataType(),true, true);
}


TEST(transferType, pluginInterface) {

    TensorVar a("a", Type(taco::Float32, {Dimension()}), taco::dense);
    TensorVar b("b", Type(taco::Float32, {Dimension()}), taco::dense);

    // should basically call a C function 
    // that can be included in header
    TransferLoad load_test("load_test", "void");
    TransferStore store_test("store_test", "void");

    TransferType("test", load_test, store_test);

    cout << load_test(makeTensorArgVarLocal(a)) << endl;
    cout << store_test(makeTensorArgVarLocal(a)) << endl;    

}