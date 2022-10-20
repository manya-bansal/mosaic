#include "taco/accelerator_notation/accelerator_notation_rewriter.h"
#include "taco/accelerator_notation/accelerator_notation_nodes.h"
#include "taco/util/collections.h"

#include <vector>

using namespace std;

namespace taco {

// class ExprRewriterStrict
AcceleratorExpr AcceleratorExprRewriterStrict::rewrite(AcceleratorExpr e) {
  if (e.defined()) {
    e.accept(this);
    e = expr;
  }
  else {
    e = AcceleratorExpr();
  }
  expr = AcceleratorExpr();
  return e;
}

// class IndexStmtRewriterStrict
AcceleratorStmt AcceleratorStmtRewriterStrict::rewrite(AcceleratorStmt s) {
  if (s.defined()) {
    s.accept(this);
    s = stmt;
  }
  else {
    s = AcceleratorStmt();
  }
  stmt = AcceleratorStmt();
  return s;
}


void AcceleratorNotationRewriter::visit(const AcceleratorAccessNode* op){
    expr = op;
}

void AcceleratorNotationRewriter::visit(const AcceleratorLiteralNode* op){
    expr = op;
}

template <class T>
AcceleratorExpr visitUnaryOp(const T *op, AcceleratorNotationRewriter *rw) {
  AcceleratorExpr a = rw->rewrite(op->a);
  if (a == op->a) {
    return op;
  }
  else {
    return new T(a);
  }
}

template <class T>
AcceleratorExpr visitBinaryOp(const T *op, AcceleratorNotationRewriter *rw) {
  AcceleratorExpr a = rw->rewrite(op->a);
  AcceleratorExpr b = rw->rewrite(op->b);
  if (a == op->a && b == op->b) {
    return op;
  }
  else {
    return new T(a, b);
  }
}

void AcceleratorNotationRewriter::visit(const AcceleratorNegNode* op){
    expr = visitUnaryOp(op, this);
}

void AcceleratorNotationRewriter::visit(const AcceleratorSqrtNode* op){
    expr = visitUnaryOp(op, this);
}

void AcceleratorNotationRewriter::visit(const AcceleratorAddNode* op){
    expr = visitBinaryOp(op, this);
}

void AcceleratorNotationRewriter::visit(const AcceleratorSubNode* op){
    expr = visitBinaryOp(op, this);
}

void AcceleratorNotationRewriter::visit(const AcceleratorMulNode* op){
    expr = visitBinaryOp(op, this);
}

void AcceleratorNotationRewriter::visit(const AcceleratorDivNode* op){
    expr = visitBinaryOp(op, this);
}

void AcceleratorNotationRewriter::visit(const AcceleratorReductionNode* op){
  AcceleratorExpr a = rewrite(op->a);
  if (a == op->a) {
    expr = op;
  }
  else {
    expr = new AcceleratorReductionNode(op->op, op->var, a);
  }
}

void AcceleratorNotationRewriter::visit(const AcceleratorDynamicIndex* op){
}


void AcceleratorNotationRewriter::visit(const AcceleratorForallNode* op){
  AcceleratorStmt s = rewrite(op->stmt);
  if (s == op->stmt) {
    stmt = op;
  }
  else {
    stmt = new AcceleratorForallNode(op->indexVar, s);
  }
}

void AcceleratorNotationRewriter::visit(const AcceleratorAssignmentNode* op){
  AcceleratorExpr rhs = rewrite(op->rhs);
  if (rhs == op->rhs) {
    stmt = op;
  }
  else {
    stmt = new AcceleratorAssignmentNode(op->lhs, rhs, op->op);
  }
}



}