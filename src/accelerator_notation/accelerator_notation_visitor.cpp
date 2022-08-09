#include "taco/accelerator_notation/accelerator_notation_nodes.h"
#include "taco/accelerator_notation/accelerator_notation_nodes.h"
#include "taco/accelerator_notation/accelerator_notation.h"

using namespace std;

namespace taco {


void AcceleratorExprVisitorStrict::visit(const AcceleratorExpr& expr) {
  cout << "here" << endl;
  expr.accept(this);
}

void AcceleratorNotationVisitor::visit(const AcceleratorAccessNode* op) {
}


}