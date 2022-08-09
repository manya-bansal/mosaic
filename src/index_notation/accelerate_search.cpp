
#include "taco/index_notation/index_notation_visitor.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/accelerator_notation/accelerate_search.h"

namespace taco {


#define CHECK_AND_ADD_EXPR                                \
do {                                                      \
    if (hasOpMatch(op, e)){                               \
        matchedPatterns.push_back(op);                    \
    }                                                     \
} while(false)

std::vector<IndexExpr> allMatchedOpPatterns(IndexStmt s, IndexExpr e){

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
        }),
        std::function<void(const WhereNode*, Matcher*)>([&](const WhereNode* op, Matcher* ctx) {
            /// do not want to explore within a where node
            return;
        }),
        std::function<void(const AccelerateNode*, Matcher*)>([&](const AccelerateNode* op, Matcher* ctx) {
            /// do not want to explore within an accelerate node
            return;
        })
    );

    return matchedPatterns;

}

bool hasOpMatch(IndexExpr e1, IndexExpr e2){

    auto OpPattern = [&](IndexExpr e, std::vector<OpTypes>& opPattern){

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
    }; 

    std::vector<OpTypes> e1Ops;
    std::vector<OpTypes> e2Ops;

    OpPattern(e1, e1Ops);
    OpPattern(e2, e2Ops);

    if (e1Ops == e2Ops){
        return true;
    }
    
    return false;

}

/// Checks whether e1 is a precise match for e2
/// this is not a symmetric relation! 
bool hasPreciseMatch(IndexExpr e1, IndexExpr e2, ArgumentMap& argumentMap){

    assert(hasOpMatch(e1, e2));

    auto PrecisePattern = [&](IndexExpr e, std::vector<EndNodeTypes>& precisePattern, std::vector<IndexExpr>& nodes) {
        match(e,
            std::function<void(const AccessNode*, Matcher*)>([&](const AccessNode* op, Matcher* ctx) {
                precisePattern.push_back(ACCESS);
                nodes.push_back(op);
            }),
            std::function<void(const LiteralNode*, Matcher*)>([&](const LiteralNode* op, Matcher* ctx) {
                precisePattern.push_back(LITERALNODE);
                nodes.push_back(op);
            }),
            std::function<void(const IndexVarNode*, Matcher*)>([&](const IndexVarNode* op, Matcher* ctx) {
                precisePattern.push_back(INDEX_VAR);
                nodes.push_back(op);
            })
        );
    };

    std::vector<EndNodeTypes> e1Pattern;
    std::vector<EndNodeTypes> e2Pattern;

    std::vector<IndexExpr> e1Nodes;
    std::vector<IndexExpr> e2Nodes;

    std::map<TensorVar, TensorVar> tensors; 
    std::map<IndexVar, IndexVar> indexVars; 

    PrecisePattern(e1, e1Pattern, e1Nodes);
    PrecisePattern(e2, e2Pattern, e2Nodes);

    // std::cout << util::join(e1Pattern) << std::endl;
    // std::cout << util::join(e2Pattern) << std::endl;
    std::cout << util::join(e1Nodes) << std::endl;
    std::cout << util::join(e2Nodes) << std::endl;

    if (e1Pattern != e2Pattern){
        return false;
    }

    for (size_t i = 0; i < e1Nodes.size() ; i++){
        switch(e1Pattern[i]){
            case ACCESS:
                {
                    auto node1 = to<AccessNode>(e1Nodes[i].ptr);
                    auto node2 = to<AccessNode>(e2Nodes[i].ptr);

                    if (node1->tensorVar.getFormat() != node2->tensorVar.getFormat()){
                        return false; 
                    }

                    if (node1->tensorVar.getType().getDataType() != node2->tensorVar.getType().getDataType()){
                        return false;
                    }

                    if (node1->tensorVar.getType().getOrder() != node2->tensorVar.getType(). getOrder()){
                        return false;
                    }   

                    int order = node1->tensorVar.getType().getOrder();
                    Shape shape1  = node1->tensorVar.getType().getShape();
                    Shape shape2  = node2->tensorVar.getType().getShape();

                    // this is what makes this check non-symmetric
                    // if e2 has dynamic dimension, but e1 has fixed dimension, 
                    // then we retrun true, but not vice-a-versa
                    for (int i = 0; i<order; i++){
                        if (shape2.getDimension(i).isVariable()){
                            continue;
                        }
                        // if it is fixed, then we need exact match
                        if (shape2.getDimension(i).isFixed()){
                            if (shape1.getDimension(i).getSize() != shape2.getDimension(i).getSize()){
                                return false;
                            }
                        }
                    }

                    if (!node1->tensorVar.hasProperties(node2->tensorVar.getProperties())){
                         return false;
                    }

                    tensors[node2->tensorVar] = node1->tensorVar;

                    for (size_t i = 0; i < node2->indexVars.size(); i++){
                        indexVars[node2->indexVars[i]] = node1->indexVars[i];
                    }

                   

                    break;
                }
            case INDEX_VAR:
                {
                    auto node1 = to<IndexVarNode>(e1Nodes[i].ptr);
                    auto node2 = to<IndexVarNode>(e2Nodes[i].ptr);

                    if (*node1 != *node2){
                        return false;
                    }
                    break;
                }
            case LITERALNODE:
                {
                    if (e1Nodes[i].getDataType() != e2Nodes[i].getDataType()){
                        return false;
                    }
                    break;
                }
            default: 
                taco_uerror << "Not reachable" << std::endl;
        }
    }

    argumentMap = ArgumentMap(tensors, indexVars, true);

    return true;

}

}
