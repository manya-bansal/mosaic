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

void AcceleratorNotationPrinter::visit(const AcceleratorDynamicIndex* op){
  // for (auto index : op->indexObject){
  //   os << index;
  // }
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



}