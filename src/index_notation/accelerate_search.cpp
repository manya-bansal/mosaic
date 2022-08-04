
#include "taco/index_notation/index_notation_visitor.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/accel_interface.h"
#include "taco/index_notation/accelerate_search.h"

namespace taco {


#define CHECK_AND_ADD_EXPR                                \
do {                                                      \
  for (auto funcDesc: accelDesc.getFuncDescriptions()){   \
        if (hasOpMatch(op, funcDesc.getExpr())){          \
            matchedPatterns.push_back(op);                \
        }                                                 \
    }                                                     \
} while(false)

std::vector<IndexExpr> allMatchedOpPatterns(IndexStmt s, AcceleratorDescription accelDesc){

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

    std::vector<OpTypes> e1Ops;
    std::vector<OpTypes> e2Ops;

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
            })
        );
    }; 


    OpPattern(e1, e1Ops);
    OpPattern(e2, e2Ops);

    if (e1Ops == e2Ops){
        return true;
    }
    
    return false;

}

}
