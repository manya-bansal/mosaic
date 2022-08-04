#ifndef ACCELERATE_SEARCH_H
#define ACCELERATE_SEARCH_H

#include "taco/index_notation/index_notation_visitor.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/accel_interface.h"

namespace taco {


enum OpTypes {ADD, SUB, NEG, MUL, SQRT, DIV, REDUX};
enum EndNodeTypes {ACCESS, LITERALNODE, INDEX_VAR};

std::vector<IndexExpr> allMatchedOpPatterns(IndexStmt s, IndexExpr e);

bool hasOpMatch(IndexExpr e1, IndexExpr e2);

bool hasPreciseMatch(IndexExpr e1, IndexExpr e2);


}

#endif