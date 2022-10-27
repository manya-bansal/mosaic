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
  void print(const DynamicStmt& expr);

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
  void visit(const DynamicIndexVarNode*);

  void visit(const DynamicEqualNode*);
  void visit(const DynamicNotEqualNode*);
  void visit(const DynamicGreaterNode*);
  void visit(const DynamicLessNode*);
  void visit(const DynamicLeqNode*);
  void visit(const DynamicGeqNode*);
  void visit(const DynamicForallNode*);
  void visit(const DynamicExistsNode*);
  void visit(const DynamicAndNode*);
  void visit(const DynamicOrNode*);

private:
  std::ostream& os;
};

class PropertyNotationPrinter : public PropertyNotationVisitorStrict {
public:
  PropertyNotationPrinter(std::ostream& os);

  void print(const PropertyExpr& expr);
  void print(const PropertyStmt& expr);

  using PropertyNotationVisitorStrict::visit;

  void visit(const PropertyTagNode*);
  void visit(const PropertyAddNode*);
  void visit(const PropertyMulNode*);
  void visit(const PropertySubNode*);
  void visit(const PropertyDivNode*);

  void visit(const PropertyAssignNode*);


private:
  std::ostream& os;
};

}
#endif
