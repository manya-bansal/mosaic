#ifndef ACCELERATOR_NOTATION_PRINTER_H
#define ACCELERATOR_NOTATION_PRINTER_H

#include <ostream>
#include "taco/accelerator_notation/accelerator_notation_visitor.h"

namespace taco {

class AcceleratorNotationPrinter : public AcceleratorNotationVisitorStrict {
public:
  AcceleratorNotationPrinter(std::ostream& os);

  void print(const AcceleratorExpr& expr);
  void print(const AcceleratorStmt& expr);

  using AcceleratorNotationVisitorStrict::visit;

  // Scalar Expressions
  void visit(const AcceleratorAccessNode*);
  void visit(const AcceleratorLiteralNode*);
  void visit(const AcceleratorNegNode*);
  void visit(const AcceleratorSqrtNode*);
  void visit(const AcceleratorAddNode*);
  void visit(const AcceleratorSubNode*);
  void visit(const AcceleratorMulNode*);
  void visit(const AcceleratorDivNode*);
  void visit(const AcceleratorReductionNode*);
  void visit(const AcceleratorDynamicIndexNode*);

  // Tensor Expressions
  void visit(const AcceleratorForallNode*);
  void visit(const AcceleratorAssignmentNode*);

private:
  std::ostream& os;

  enum class Precedence {
    ACCESS = 2,
    FUNC = 2,
    CAST = 2,
    REDUCTION = 2,
    NEG = 3,
    MUL = 5,
    DIV = 5,
    ADD = 6,
    SUB = 6,
    TOP = 20
  };
  Precedence parentPrecedence;

  template <typename Node> void visitAcceleratedBinary(Node op, Precedence p);
  template <typename Node> void visitImmediate(Node op);
};

class DynamicNotationPrinter : public DynamicNotationVisitorStrict {
public:
  DynamicNotationPrinter(std::ostream& os);

  void print(const DynamicExpr& expr);
  // void print(const AcceleratorStmt& expr);

  using DynamicNotationVisitorStrict::visit;

  void visit(const DynamicIndexIteratorNode*);
  void visit(const DynamicIndexAccessNode*);
  void visit(const DynamicLiteralNode*);
  void visit(const DynamicIndexLenNode*);
  void visit(const DynamicIndexMulInternalNode*);
  void visit(const DynamicAddNode*);
  void visit(const DynamicSubNode*);
  void visit(const DynamicMulNode*);
  void visit(const DynamicDivNode*);
  void visit(const DynamicModNode*);

private:
  std::ostream& os;
};

}
#endif
