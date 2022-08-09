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

void AcceleratorNotationPrinter::visit(const AcceleratorAccessNode* op) { 
  os << op->tensorObject.getName();
  if (op->isAccessingStructure) {
    os << "_struct";
  }
  if (op->indexVars.size() > 0) {
    os << "(" << util::join(op->indexVars,",") << ")";
  }
}

}