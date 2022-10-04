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
#include "taco/accelerator_notation/accelerator_notation_visitor.h"
#include "taco/accelerator_notation/accelerator_notation_nodes.h"
#include "taco/accelerator_notation/interface_schedular.h"
#include "taco/lower/lower.h"
#include "taco/ir_tags.h"
#include "taco/error/error_messages.h"

#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/test_interface.h"
#include "taco/accelerator_interface/tile_interface.h"


using namespace taco;

int numOperators(const AcceleratorStmt& stmt){

    int numOps = 0;

    acceleratorMatch(stmt,
        std::function<void(const AcceleratorAddNode*, AcceleratorMatcher*)>([&](const AcceleratorAddNode* op, AcceleratorMatcher* ctx) {
           numOps++;
           ctx->acceleratorMatch(op->a);
           ctx->acceleratorMatch(op->b);
        }),
        std::function<void(const AcceleratorSubNode*, AcceleratorMatcher*)>([&](const AcceleratorSubNode* op, AcceleratorMatcher* ctx) {
           numOps++;
           ctx->acceleratorMatch(op->a);
           ctx->acceleratorMatch(op->b);
        }),
        std::function<void(const AcceleratorNegNode*, AcceleratorMatcher*)>([&](const AcceleratorNegNode* op, AcceleratorMatcher* ctx) {
           numOps++;
           ctx->acceleratorMatch(op->a);
        }),
        std::function<void(const AcceleratorMulNode*, AcceleratorMatcher*)>([&](const AcceleratorMulNode* op, AcceleratorMatcher* ctx) {
           numOps++;
           ctx->acceleratorMatch(op->a);
           ctx->acceleratorMatch(op->b);
        }),
        std::function<void(const AcceleratorSqrtNode*, AcceleratorMatcher*)>([&](const AcceleratorSqrtNode* op, AcceleratorMatcher* ctx) {
           numOps++;
           ctx->acceleratorMatch(op->a);
        }),
        std::function<void(const AcceleratorDivNode*, AcceleratorMatcher*)>([&](const AcceleratorDivNode* op, AcceleratorMatcher* ctx) {
           numOps++;
           ctx->acceleratorMatch(op->a);
           ctx->acceleratorMatch(op->b);
        })
    );

    return numOps;
}

bool customCompare(const Pile& a, const Pile&b){
    return numOperators(a.getTargetStmt()) < numOperators(b.getTargetStmt()); 
}

TEST(iSchedular, testDummp){



    Pile pile1(Saxpy().getStmt());

    Piles piles(customCompare);

    piles.feed(pile1);

    while(piles.empty()==false){
      cout<< piles.get().getTargetStmt() <<"\n";
      piles.pop();
    }

}