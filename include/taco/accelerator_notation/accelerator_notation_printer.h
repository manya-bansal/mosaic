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
  //   void visit(const SqrtNode*);
  void visit(const AcceleratorAddNode*);

//   void visit(const SubNode*);
//   void visit(const MulNode*);
//   void visit(const DivNode*);
//   void visit(const CastNode*);
//   void visit(const CallNode*);
//   void visit(const CallIntrinsicNode*);
//   void visit(const ReductionNode*);
//   void visit(const IndexVarNode*);

//   // Tensor Expressions
  void visit(const AcceleratorAssignmentNode*);
//   void visit(const YieldNode*);
//   void visit(const ForallNode*);
//   void visit(const WhereNode*);
//   void visit(const AccelerateNode*);
//   void visit(const MultiNode*);
//   void visit(const SequenceNode*);
//   void visit(const AssembleNode*);
//   void visit(const SuchThatNode*);

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

}
#endif
