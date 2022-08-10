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

void AcceleratorNotationPrinter::visit(const AcceleratorAssignmentNode* op) {
  op->lhs.accept(this);
  // TODO: Right now, only supports = 
  os << " = ";
  op->rhs.accept(this);
}

}