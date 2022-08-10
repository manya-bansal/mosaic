#include "taco/accelerator_notation/accelerator_notation_nodes.h"
#include "taco/accelerator_notation/accelerator_notation_nodes.h"
#include "taco/accelerator_notation/accelerator_notation.h"

using namespace std;

namespace taco {


void AcceleratorExprVisitorStrict::visit(const AcceleratorExpr& expr) {
  expr.accept(this);
}

void AcceleratorNotationVisitor::visit(const AcceleratorAccessNode* op) {
}

void AcceleratorNotationVisitor::visit(const AcceleratorLiteralNode* op) {
}

void AcceleratorStmtVisitorStrict::visit(const AcceleratorStmt& expr) {
  expr.accept(this);
}

void AcceleratorNotationVisitor::visit(const AcceleratorAssignmentNode* op) {
  op->rhs.accept(this);
}

void AcceleratorNotationVisitor::visit(const AcceleratorUnaryExprNode* op) {
  op->a.accept(this);
}

void AcceleratorNotationVisitor::visit(const AcceleratorBinaryExprNode* op) {
  op->a.accept(this);
  op->b.accept(this);
}

void AcceleratorNotationVisitor::visit(const AcceleratorNegNode* op) {
  visit(static_cast<const AcceleratorUnaryExprNode*>(op));
}

void AcceleratorNotationVisitor::visit(const AcceleratorSqrtNode* op) {
  visit(static_cast<const AcceleratorUnaryExprNode*>(op));
}

void AcceleratorNotationVisitor::visit(const AcceleratorAddNode* op) {
  visit(static_cast<const AcceleratorBinaryExprNode*>(op));
}

void AcceleratorNotationVisitor::visit(const AcceleratorSubNode* op) {
  visit(static_cast<const AcceleratorBinaryExprNode*>(op));
}
void AcceleratorNotationVisitor::visit(const AcceleratorMulNode* op) {
  visit(static_cast<const AcceleratorBinaryExprNode*>(op));
}
void AcceleratorNotationVisitor::visit(const AcceleratorDivNode* op) {
  visit(static_cast<const AcceleratorBinaryExprNode*>(op));
}

void AcceleratorNotationVisitor::visit(const AcceleratorReductionNode* op) {
  op->a.accept(this);
}

void AcceleratorNotationVisitor::visit(const AcceleratorForallNode* op) {
  op->stmt.accept(this);
}


}