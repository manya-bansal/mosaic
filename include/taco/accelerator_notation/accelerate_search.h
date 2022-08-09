#ifndef ACCELERATE_SEARCH_H
#define ACCELERATE_SEARCH_H

#include "taco/index_notation/index_notation_visitor.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_notation/accel_interface.h"

namespace taco {


enum OpTypes {ADD, SUB, NEG, MUL, SQRT, DIV, REDUX};
enum EndNodeTypes {ACCESS, LITERALNODE, INDEX_VAR};

/// Describes how to go go from 
/// args written in the pligin 
/// to the args given 
/// in the concrete expression
struct ArgumentMap{
    ArgumentMap() = default;
    ArgumentMap(const std::map<TensorVar, TensorVar>& tensors,
            const std::map<IndexVar, IndexVar>& indexVars, 
            bool possible) : tensors(tensors), indexVars(indexVars),
            possible(possible) {}

    std::map<TensorVar, TensorVar> tensors; 
    std::map<IndexVar, IndexVar> indexVars; 
    bool possible; 

};

std::vector<IndexExpr> allMatchedOpPatterns(IndexStmt s, IndexExpr e);

bool hasOpMatch(IndexExpr e1, IndexExpr e2);

bool hasPreciseMatch(IndexExpr e1, IndexExpr e2, ArgumentMap& argumentMap);


}

#endif