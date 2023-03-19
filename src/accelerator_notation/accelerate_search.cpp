#include <functional>

#include "taco/index_notation/index_notation_visitor.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/accelerator_notation/accelerator_notation_nodes.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/accelerator_notation/accelerate_search.h"
#include "taco/accelerator_notation/accelerator_notation_visitor.h"

namespace taco {


#define CHECK_AND_ADD_EXPR                                \
do {                                                      \
    if (hasOpMatch(op, e)){                               \
        matchedPatterns.push_back(op);                    \
    }                                                     \
} while(false)

std::vector<IndexExpr> allMatchedOpPatterns(IndexExpr s, AcceleratorExpr e){

    std::vector<IndexExpr> matchedPatterns;

    match(s,
        std::function<void(const AddNode*, Matcher*)>([&](const AddNode* op, Matcher* ctx) {
            CHECK_AND_ADD_EXPR;   
            ctx->match(op->a);
            ctx->match(op->b);
        }),
        std::function<void(const MulNode*, Matcher*)>([&](const MulNode* op, Matcher* ctx) {
            CHECK_AND_ADD_EXPR;
            ctx->match(op->a);
            ctx->match(op->b);
        }),
        std::function<void(const SubNode*, Matcher*)>([&](const SubNode* op, Matcher* ctx) {
            CHECK_AND_ADD_EXPR;
            ctx->match(op->a);
            ctx->match(op->b);
        }),
        std::function<void(const NegNode*, Matcher*)>([&](const NegNode* op, Matcher* ctx) {
            CHECK_AND_ADD_EXPR;
            ctx->match(op->a);
        }),
        std::function<void(const SqrtNode*, Matcher*)>([&](const SqrtNode* op, Matcher* ctx) {
            CHECK_AND_ADD_EXPR;
            ctx->match(op->a);
        }),
        std::function<void(const DivNode*, Matcher*)>([&](const DivNode* op, Matcher* ctx) {
            CHECK_AND_ADD_EXPR;
            ctx->match(op->a);
            ctx->match(op->b);
        }),
        std::function<void(const ReductionNode*, Matcher*)>([&](const ReductionNode* op, Matcher* ctx) {
            CHECK_AND_ADD_EXPR;
            ctx->match(op->a);
        })
    );

    return matchedPatterns;

}

static std::vector<OpTypes> getOpPattern(IndexExpr e){

    std::vector<OpTypes> opPattern;

    match(e,
        std::function<void(const AddNode*, Matcher*)>([&](const AddNode* op, Matcher* ctx) {
            opPattern.push_back(ADD);
            ctx->match(op->a);
            ctx->match(op->b);
        }),
        std::function<void(const SubNode*, Matcher*)>([&](const SubNode* op, Matcher* ctx) {
            opPattern.push_back(SUB);
            ctx->match(op->a);
            ctx->match(op->b);
        }),
        std::function<void(const NegNode*, Matcher*)>([&](const NegNode* op, Matcher* ctx) {
            opPattern.push_back(NEG);
            ctx->match(op->a);
        }),
        std::function<void(const MulNode*, Matcher*)>([&](const MulNode* op, Matcher* ctx) {
            opPattern.push_back(MUL);
            ctx->match(op->a);
            ctx->match(op->b);
        }),
        std::function<void(const SqrtNode*, Matcher*)>([&](const SqrtNode* op, Matcher* ctx) {
            opPattern.push_back(SQRT);
            ctx->match(op->a);
        }),
        std::function<void(const DivNode*, Matcher*)>([&](const DivNode* op, Matcher* ctx) {
            opPattern.push_back(DIV);
            ctx->match(op->a);
            ctx->match(op->b);
        }),
        std::function<void(const ReductionNode*, Matcher*)>([&](const ReductionNode* op, Matcher* ctx) {
            opPattern.push_back(REDUX);
            ctx->match(op->a);
        })
    );

    return opPattern;

}

static std::vector<OpTypes> getOpPattern(AcceleratorExpr e){

    std::vector<OpTypes> opPattern;

    acceleratorMatch(e,
        std::function<void(const AcceleratorAddNode*, AcceleratorMatcher*)>([&](const AcceleratorAddNode* op, AcceleratorMatcher* ctx) {
            opPattern.push_back(ADD);
            ctx->acceleratorMatch(op->a);
            ctx->acceleratorMatch(op->b);
        }),
        std::function<void(const AcceleratorSubNode*, AcceleratorMatcher*)>([&](const AcceleratorSubNode* op, AcceleratorMatcher* ctx) {
            opPattern.push_back(SUB);
            ctx->acceleratorMatch(op->a);
            ctx->acceleratorMatch(op->b);
        }),
        std::function<void(const AcceleratorNegNode*, AcceleratorMatcher*)>([&](const AcceleratorNegNode* op, AcceleratorMatcher* ctx) {
            opPattern.push_back(NEG);
            ctx->acceleratorMatch(op->a);
        }),
        std::function<void(const AcceleratorMulNode*, AcceleratorMatcher*)>([&](const AcceleratorMulNode* op, AcceleratorMatcher* ctx) {
            opPattern.push_back(MUL);
            ctx->acceleratorMatch(op->a);
            ctx->acceleratorMatch(op->b);
        }),
        std::function<void(const AcceleratorSqrtNode*, AcceleratorMatcher*)>([&](const AcceleratorSqrtNode* op, AcceleratorMatcher* ctx) {
            opPattern.push_back(SQRT);
            ctx->acceleratorMatch(op->a);
        }),
        std::function<void(const AcceleratorDivNode*, AcceleratorMatcher*)>([&](const AcceleratorDivNode* op, AcceleratorMatcher* ctx) {
            opPattern.push_back(DIV);
            ctx->acceleratorMatch(op->a);
            ctx->acceleratorMatch(op->b);
        }),
        std::function<void(const AcceleratorReductionNode*, AcceleratorMatcher*)>([&](const AcceleratorReductionNode* op, AcceleratorMatcher* ctx) {
            opPattern.push_back(REDUX);
            ctx->acceleratorMatch(op->a);
        })
    );

    return opPattern;

}

bool hasOpMatch(IndexExpr e1, AcceleratorExpr e2){

    if (getOpPattern(e1) == getOpPattern(e2)){
        return true;
    }
    return false;
}

/// Checks whether e1 is a precise match for e2
/// this does not check whether the dimesnions match
/// that matching can be forced using a tiling op!
ArgumentMap hasPreciseMatch(IndexExpr e1, AcceleratorExpr e2){

    // std::cout << e2 << std::endl;
    // std::cout << util::join(getOpPattern(e2)) << std::endl;
    // std::cout << util::join(getOpPattern(e1)) << std::endl;

    if (!hasOpMatch(e1, e2)) {
        return ArgumentMap(false);
    }

    auto PrecisePatternIndexExpr = [&](IndexExpr e, std::vector<EndNodeTypes>& precisePattern, std::vector<IndexExpr>& nodes) {
        match(e,
            std::function<void(const AccessNode*, Matcher*)>([&](const AccessNode* op, Matcher* ctx) {
                precisePattern.push_back(ACCESS);
                nodes.push_back(op);
            }),
            std::function<void(const LiteralNode*, Matcher*)>([&](const LiteralNode* op, Matcher* ctx) {
                precisePattern.push_back(LITERALNODE);
                nodes.push_back(op);
            })
        );
    };

    auto PrecisePatternAccelExpr = [&](AcceleratorExpr e, std::vector<EndNodeTypes>& precisePattern, std::vector<AcceleratorExpr>& nodes) {
        acceleratorMatch(e,
            std::function<void(const AcceleratorAccessNode*, AcceleratorMatcher*)>([&](const AcceleratorAccessNode* op, AcceleratorMatcher* ctx) {
                precisePattern.push_back(ACCESS);
                nodes.push_back(op);
            }),
            std::function<void(const AcceleratorLiteralNode*, AcceleratorMatcher*)>([&](const AcceleratorLiteralNode* op, AcceleratorMatcher* ctx) {
                precisePattern.push_back(LITERALNODE);
                nodes.push_back(op);
            })
        );
    };

    std::vector<EndNodeTypes> e1Pattern;
    std::vector<EndNodeTypes> e2Pattern;

    std::vector<IndexExpr> e1Nodes;
    std::vector<AcceleratorExpr> e2Nodes;

    std::map<TensorObject, TensorVar> tensors; 
    std::map<IndexVar, IndexVar> indexVars; 

    PrecisePatternIndexExpr(e1, e1Pattern, e1Nodes);
    PrecisePatternAccelExpr(e2, e2Pattern, e2Nodes);

    if (e1Pattern != e2Pattern){
        return ArgumentMap(false);
    }

    for (size_t i = 0; i < e1Nodes.size() ; i++){
        switch(e1Pattern[i]){
            case ACCESS:
                {
                    auto node1 = to<AccessNode>(e1Nodes[i].ptr);
                    auto node2 = to<AcceleratorAccessNode>(e2Nodes[i].ptr);

                    if (node1->tensorVar.getFormat() != node2->tensorObject.getFormat()){
                        return  ArgumentMap(false); 
                    }

                    if (node1->tensorVar.getType().getDataType() != node2->tensorObject.getType().getDataType()){
                        return  ArgumentMap(false);
                    }

                    if (node1->tensorVar.getType().getOrder() != node2->tensorObject.getType(). getOrder()){
                        return  ArgumentMap(false);
                    }   

                    // int order = node1->tensorVar.getType().getOrder();
                    // Shape shape1  = node1->tensorVar.getType().getShape();
                    // Shape shape2  = node2->tensorObject.getType().getShape();


                    // for (int i = 0; i<order; i++){
                    //     if (shape2.getDimension(i).isVariable()){
                    //         continue;
                    //     }
                    //     // if it is fixed, then we need exact match
                    //     if (shape2.getDimension(i).isFixed()){
                    //         if (shape1.getDimension(i).getSize() != shape2.getDimension(i).getSize()){
                    //             return false;
                    //         }
                    //     }
                    // }

                    if (!node1->tensorVar.hasProperties(node2->tensorObject.getProperties())){
                         return false;
                    }

                    tensors[node2->tensorObject] = node1->tensorVar;

                    for (size_t i = 0; i < node2->indexVars.size(); i++){
                        indexVars[node2->indexVars[i]] = node1->indexVars[i];
                    }

                    break;
                }

            case LITERALNODE:
                {
                    if (e1Nodes[i].getDataType() != e2Nodes[i].getDataType()){
                        return ArgumentMap(false);
                    }
                    break;
                }
            default: 
                taco_uerror << "Not reachable" << std::endl;
        }
    }

    return ArgumentMap(tensors, indexVars, true);

}


bool matches(std::set<IndexVar> reduxVar, std::set<IndexVar> reduxVarsRef,  std::set<DynamicOrder> reduxOrderRef){
  return true;
}

bool isSameReduxPattern(AcceleratorStmt refStmt, IndexStmt stmt){
  std::vector<IndexVar> reduxVars = getReductionVars(stmt);
  return true;
}

bool matches(AcceleratorStmt refStmt, IndexStmt stmt){
  isSameReduxPattern(refStmt, stmt);
  return true;
}

std::map<IndexExpr, AcceleratorExpr> getMatchingTensors(IndexExpr e1, AcceleratorExpr e2){

    auto getTensorsIndexExpr = [&](IndexExpr e, std::vector<EndNodeTypes>& precisePattern, std::vector<IndexExpr>& nodes) {
        match(e,
            std::function<void(const AccessNode*, Matcher*)>([&](const AccessNode* op, Matcher* ctx) {
                precisePattern.push_back(ACCESS);
                nodes.push_back(op);
            }),
            std::function<void(const LiteralNode*, Matcher*)>([&](const LiteralNode* op, Matcher* ctx) {
                precisePattern.push_back(LITERALNODE);
                nodes.push_back(op);
            })
        );
    };

    auto getTensorsAccelExpr = [&](AcceleratorExpr e, std::vector<EndNodeTypes>& precisePattern, std::vector<AcceleratorExpr>& nodes) {
        acceleratorMatch(e,
            std::function<void(const AcceleratorAccessNode*, AcceleratorMatcher*)>([&](const AcceleratorAccessNode* op, AcceleratorMatcher* ctx) {
                precisePattern.push_back(ACCESS);
                nodes.push_back(op);
            }),
            std::function<void(const AcceleratorLiteralNode*, AcceleratorMatcher*)>([&](const AcceleratorLiteralNode* op, AcceleratorMatcher* ctx) {
                precisePattern.push_back(LITERALNODE);
                nodes.push_back(op);
            })
        );
    };

    std::vector<EndNodeTypes> e1Pattern;
    std::vector<EndNodeTypes> e2Pattern;

    std::vector<IndexExpr> e1Nodes;
    std::vector<AcceleratorExpr> e2Nodes;

    std::map<IndexExpr, AcceleratorExpr> tensors; 

    getTensorsIndexExpr(e1, e1Pattern, e1Nodes);
    getTensorsAccelExpr(e2, e2Pattern, e2Nodes);

    for (size_t i = 0; i < e1Nodes.size() ; i++){
        tensors[e1Nodes[i]] = e2Nodes[i];
    }

    return tensors;

}

}