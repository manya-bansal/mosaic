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

void AcceleratorNotationVisitor::visit(const AcceleratorDynamicIndexNode*){
}

void AcceleratorNotationVisitor::visit(const AcceleratorForallNode* op) {
  op->stmt.accept(this);
}

void DynamicExprVisitorStrict::visit(const DynamicExpr& expr){
  expr.accept(this);
}

void DynamicStmtVisitorStrict::visit(const DynamicStmt& stmt){
  stmt.accept(this);
}

void DynamicNotationVisitor::visit(const DynamicIndexIteratorNode*){
}
void DynamicNotationVisitor::visit(const DynamicIndexAccessNode*){
}
void DynamicNotationVisitor::visit(const DynamicLiteralNode*){
}
void DynamicNotationVisitor::visit(const DynamicIndexLenNode* op){
}
void DynamicNotationVisitor::visit(const DynamicIndexMulInternalNode* op){
}
void DynamicNotationVisitor::visit(const DynamicAddNode* op){
  op->a.accept(this);
  op->b.accept(this);
}
void DynamicNotationVisitor::visit(const DynamicSubNode* op){
  op->a.accept(this);
  op->b.accept(this);
}
void DynamicNotationVisitor::visit(const DynamicMulNode* op){
  op->a.accept(this);
  op->b.accept(this);
}
void DynamicNotationVisitor::visit(const DynamicDivNode* op){
  op->a.accept(this);
  op->b.accept(this);
}
void DynamicNotationVisitor::visit(const DynamicModNode* op){
  op->a.accept(this);
  op->b.accept(this);
}
void DynamicNotationVisitor::visit(const DynamicIndexVarNode*){
}
void DynamicNotationVisitor::visit(const DynamicEqualNode* op){
  op->a.accept(this);
  op->b.accept(this);
}
void DynamicNotationVisitor::visit(const DynamicNotEqualNode* op){
  op->a.accept(this);
  op->b.accept(this);
}
void DynamicNotationVisitor::visit(const DynamicGreaterNode* op){
  op->a.accept(this);
  op->b.accept(this);
}
void DynamicNotationVisitor::visit(const DynamicLessNode* op){
  op->a.accept(this);
  op->b.accept(this);
}
void DynamicNotationVisitor::visit(const DynamicLeqNode* op){
  op->a.accept(this);
  op->b.accept(this);
}
void DynamicNotationVisitor::visit(const DynamicGeqNode* op){
  op->a.accept(this);
  op->b.accept(this);
}
void DynamicNotationVisitor::visit(const DynamicForallNode* op){
  op->stmt.accept(this);
}
void DynamicNotationVisitor::visit(const DynamicExistsNode* op){
  op->stmt.accept(this);
}

void DynamicNotationVisitor::visit(const DynamicAndNode* op){
  op->a.accept(this);
  op->b.accept(this);
}

void DynamicNotationVisitor::visit(const DynamicOrNode* op){
  op->a.accept(this);
  op->b.accept(this);
}

void PropertyExprVisitorStrict::visit(const PropertyExpr& expr) {
  expr.accept(this);
}

void PropertyStmtVisitorStrict::visit(const PropertyStmt& expr) {
  expr.accept(this);
}

void PropertyExprVisitorStrict::visit(const PropertyTagNode*){
}

}