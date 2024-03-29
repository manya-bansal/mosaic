#include "taco/accelerator_notation/accelerator_notation_printer.h"
#include "taco/accelerator_notation/accelerator_notation_nodes.h"
#include "taco/accelerator_notation/accelerator_notation.h"

using namespace std;

namespace taco {

AcceleratorNotationPrinter::AcceleratorNotationPrinter(std::ostream& os) : os(os) {
}

void AcceleratorNotationPrinter::print(const AcceleratorExpr& expr) {
  parentPrecedence = Precedence::TOP;
  expr.accept(this);
}

void AcceleratorNotationPrinter::print(const AcceleratorStmt& expr) {
  parentPrecedence = Precedence::TOP;
  expr.accept(this);
}

void AcceleratorNotationPrinter::visit(const AcceleratorAccessNode* op) { 
  os << op->tensorObject.getName();
  if (op->isAccessingStructure) {
    os << "_struct";
  }
  if (op->indexVars.size() > 0) {
    os << "(" << util::join(op->indexVars,",") << ")";
  }
}

void AcceleratorNotationPrinter::visit(const AcceleratorLiteralNode* op) { 
    switch (op->getDataType().getKind()) {
    case Datatype::Bool:
      os << op->getVal<bool>();
      break;
    case Datatype::UInt8:
      os << op->getVal<uint8_t>();
      break;
    case Datatype::UInt16:
      os << op->getVal<uint16_t>();
      break;
    case Datatype::UInt32:
      os << op->getVal<uint32_t>();
      break;
    case Datatype::UInt64:
      os << op->getVal<uint64_t>();
      break;
    case Datatype::UInt128:
      taco_not_supported_yet;
      break;
    case Datatype::Int8:
      os << op->getVal<int8_t>();
      break;
    case Datatype::Int16:
      os << op->getVal<int16_t>();
      break;
    case Datatype::Int32:
      os << op->getVal<int32_t>();
      break;
    case Datatype::Int64:
      os << op->getVal<int64_t>();
      break;
    case Datatype::Int128:
      taco_not_supported_yet;
      break;
    case Datatype::Float32:
      os << op->getVal<float>();
      break;
    case Datatype::Float64:
      os << op->getVal<double>();
      break;
    case Datatype::Complex64:
      os << op->getVal<std::complex<float>>();
      break;
    case Datatype::Complex128:
      os << op->getVal<std::complex<double>>();
      break;
    case Datatype::Undefined:
      break;
  }
}

void AcceleratorNotationPrinter::visit(const AcceleratorNegNode* op) {
  Precedence precedence = Precedence::NEG;
  bool parenthesize =  precedence > parentPrecedence;
  parentPrecedence = precedence;
  if(op->getDataType().isBool()) {
    os << "!";
  } else {
    os << "-";
  }

  if (parenthesize) {
    os << "(";
  }
  op->a.accept(this);
  if (parenthesize) {
    os << ")";
  }
}

void AcceleratorNotationPrinter::visit(const AcceleratorSqrtNode* op) {
  parentPrecedence = Precedence::FUNC;
  os << "sqrt";
  os << "(";
  op->a.accept(this);
  os << ")";
}

template <typename Node>
void AcceleratorNotationPrinter::visitAcceleratedBinary(Node op, Precedence precedence) {
  bool parenthesize =  precedence > parentPrecedence;
  if (parenthesize) {
    os << "(";
  }
  parentPrecedence = precedence;
  op->a.accept(this);
  os << " " << op->getOperatorString() << " ";
  parentPrecedence = precedence;
  op->b.accept(this);
  if (parenthesize) {
    os << ")";
  }
}

void AcceleratorNotationPrinter::visit(const AcceleratorAddNode* op){
  visitAcceleratedBinary(op, Precedence::ADD);
}

void AcceleratorNotationPrinter::visit(const AcceleratorSubNode* op){
  visitAcceleratedBinary(op, Precedence::SUB);
}

void AcceleratorNotationPrinter::visit(const AcceleratorMulNode* op){
  visitAcceleratedBinary(op, Precedence::MUL);
}

void AcceleratorNotationPrinter::visit(const AcceleratorDivNode* op){
  visitAcceleratedBinary(op, Precedence::DIV);
}

void AcceleratorNotationPrinter::visit(const AcceleratorReductionNode* op) {

  struct ReductionName : AcceleratorNotationVisitor {
    std::string reductionName;
    std::string get(AcceleratorExpr expr) {
      expr.accept(this);
      return reductionName;
    }
    using AcceleratorNotationVisitor::visit;
    void visit(const AcceleratorAddNode* node) {
      reductionName = "sum";
    }
    void visit(const  AcceleratorMulNode* node) {
      reductionName = "product";
    }
    void visit(const  AcceleratorBinaryExprNode* node) {
      reductionName = "reduction(" + node->getOperatorString() + ")";
    }
  };

  parentPrecedence = Precedence::REDUCTION;
  os << ReductionName().get(op->op) << "(" << op->var << ", ";
  op->a.accept(this);
  os << ")";
}

void AcceleratorNotationPrinter::visit(const AcceleratorDynamicIndexNode* op){
  os << op->t.getName() << "(";
  for (auto index : op->indexObject){
    os << index << " ";
  }
  os << ")";
}


void AcceleratorNotationPrinter::visit(const AcceleratorForallNode* op) {
  os << "forall(" << op->indexVar << ", ";
  op->stmt.accept(this);
  os << ")";
}

void AcceleratorNotationPrinter::visit(const AcceleratorAssignmentNode* op) {

  struct OperatorName : AcceleratorNotationVisitor {
    using AcceleratorNotationVisitor::visit;
    std::string operatorName;
    std::string get(AcceleratorExpr expr) {
      if (!expr.defined()) return "";
      expr.accept(this);
      return operatorName;
    }
    void visit(const AcceleratorBinaryExprNode* node){
      operatorName = node->getOperatorString();
    }

  };
  op->lhs.accept(this);
  // TODO: Right now, only supports = 
  os << " " << OperatorName().get(op->op) << "= ";
  op->rhs.accept(this);
}

DynamicNotationPrinter::DynamicNotationPrinter(std::ostream& os) : os(os){
}

void DynamicNotationPrinter::print(const DynamicExpr& expr) {
  expr.accept(this);
}

void DynamicNotationPrinter::print(const DynamicStmt& expr) {
  expr.accept(this);
}

void DynamicNotationPrinter::visit(const DynamicIndexIteratorNode* op){
   os << "iterate(" << op->dynamicOrder.getName() << ")";
}

void DynamicNotationPrinter::visit(const DynamicIndexAccessNode* op){
   os << op->it.getDynamicOrder().getName() << "(" << "access" <<")";
}

void DynamicNotationPrinter::visit(const DynamicLiteralNode* op){
  os << op->num;
}

void DynamicNotationPrinter::visit(const DynamicIndexLenNode* op){
  os << "|" << op->dynamicOrder.getName() << "|";
}

void DynamicNotationPrinter::visit(const DynamicIndexMulInternalNode* op){
  os << "(*" << op->dynamicOrder.getName() << ")";
}

void DynamicNotationPrinter::visit(const DynamicAddNode* op){
  op->a.accept(this);
  os << "+";
  op->b.accept(this);
}

void DynamicNotationPrinter::visit(const DynamicSubNode* op){
  op->a.accept(this);
  os << "-";
  op->b.accept(this);
}

void DynamicNotationPrinter::visit(const DynamicMulNode* op){
  op->a.accept(this);
  os << "*";
  op->b.accept(this);
}

void DynamicNotationPrinter::visit(const DynamicDivNode* op){
  op->a.accept(this);
  os << "/";
  op->b.accept(this);
}

void DynamicNotationPrinter::visit(const DynamicModNode* op){
  op->a.accept(this);
  os << "%";
  op->b.accept(this);
}

void DynamicNotationPrinter::visit(const DynamicEqualNode* op){
  op->a.accept(this);
  os << "==";
  op->b.accept(this);
}

void DynamicNotationPrinter::visit(const DynamicNotEqualNode* op){
  op->a.accept(this);
  os << "!=";
  op->b.accept(this);
}

void DynamicNotationPrinter::visit(const DynamicGreaterNode* op){
  op->a.accept(this);
  os << ">";
  op->b.accept(this);
}

void DynamicNotationPrinter::visit(const DynamicLessNode* op){
  op->a.accept(this);
  os << "<";
  op->b.accept(this);
}

void DynamicNotationPrinter::visit(const DynamicGeqNode* op){
  op->a.accept(this);
  os << ">=";
  op->b.accept(this);
}

void DynamicNotationPrinter::visit(const DynamicLeqNode* op){
  op->a.accept(this);
  os << "<=";
  op->b.accept(this);
}

void DynamicNotationPrinter::visit(const DynamicForallNode* op){
  os << "forall(";
  os << op->it << ", ";
  op->stmt.accept(this);
  os << ")";
}

void DynamicNotationPrinter::visit(const DynamicExistsNode* op){
  os << "exists(";
  os << op->it << ", ";
  op->stmt.accept(this);
  os << ")";
}

void DynamicNotationPrinter::visit(const DynamicIndexVarNode* op){
  os << op->i;
}

void DynamicNotationPrinter::visit(const DynamicAndNode* op){
  op->a.accept(this);
  os << "&&";
  op->b.accept(this);
}

void DynamicNotationPrinter::visit(const DynamicOrNode* op){
  op->a.accept(this);
  os << "&&";
  op->b.accept(this);
}

PropertyNotationPrinter::PropertyNotationPrinter(std::ostream& os) : os(os){
}

void PropertyNotationPrinter::print(const PropertyExpr& expr) {
  expr.accept(this);
}

void PropertyNotationPrinter::print(const PropertyStmt& expr) {
  expr.accept(this);
}

void PropertyNotationPrinter::visit(const PropertyTagNode* op){
  os << op->property;
}

void PropertyNotationPrinter::visit(const PropertyAddNode* op){
  os << op->a  << " + " << op->b;
}

void PropertyNotationPrinter::visit(const PropertySubNode* op){
  os << op->a  << " - " << op->b;
}

void PropertyNotationPrinter::visit(const PropertyMulNode* op){
  os << op->a  << " * " << op->b;
}

void PropertyNotationPrinter::visit(const PropertyDivNode* op){
  os << op->a  << " / " << op->b;
}

void PropertyNotationPrinter::visit(const PropertyAssignNode* op){
  os << op->lhs  << " = " << op->rhs;
}

}