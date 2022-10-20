#ifndef ACCELERATOR_NOTATION_VISITOR_H
#define ACCELERATOR_NOTATION_VISITOR_H

#include <vector>
#include <functional>
#include "taco/error.h"

namespace taco {

class AcceleratorExpr;
class AcceleratorStmt;

struct AcceleratorAccessNode;
struct AcceleratorLiteralNode;
struct AcceleratorNegNode;
struct AcceleratorSqrtNode;
struct AcceleratorAddNode;
struct AcceleratorSubNode;
struct AcceleratorMulNode;
struct AcceleratorDivNode;
struct AcceleratorReductionNode;
struct AcceleratorDynamicIndex;

struct AcceleratorBinaryExprNode;
struct AcceleratorUnaryExprNode;

struct AcceleratorAssignmentNode;
struct AcceleratorForallNode;

class AcceleratorExprVisitorStrict {
    public:
        virtual ~AcceleratorExprVisitorStrict() = default;

        void visit(const AcceleratorExpr&);

        virtual void visit(const AcceleratorAccessNode*) = 0;
        virtual void visit(const AcceleratorLiteralNode*) = 0;
        virtual void visit(const AcceleratorNegNode*) = 0;
        virtual void visit(const AcceleratorSqrtNode*) = 0;
        virtual void visit(const AcceleratorAddNode*) = 0;
        virtual void visit(const AcceleratorSubNode*) = 0;
        virtual void visit(const AcceleratorDivNode*) = 0;
        virtual void visit(const AcceleratorMulNode*) = 0;
        virtual void visit(const AcceleratorReductionNode*) = 0;
        virtual void visit(const AcceleratorDynamicIndex*) = 0;

};

class AcceleratorStmtVisitorStrict {
    public:
        virtual ~AcceleratorStmtVisitorStrict() = default;

        void visit(const AcceleratorStmt&);

        virtual void visit(const AcceleratorAssignmentNode*) = 0;
        virtual void visit(const AcceleratorForallNode*) = 0; 
 };

/// Visit nodes in index notation
class AcceleratorNotationVisitorStrict : public AcceleratorExprVisitorStrict, 
                                         public AcceleratorStmtVisitorStrict {
    public:
        virtual ~AcceleratorNotationVisitorStrict() = default;

        using AcceleratorExprVisitorStrict::visit;
        using AcceleratorStmtVisitorStrict::visit;
};

class AcceleratorNotationVisitor : public AcceleratorNotationVisitorStrict {
public:
  virtual ~AcceleratorNotationVisitor() = default;

  using AcceleratorExprVisitorStrict::visit;

  virtual void visit(const AcceleratorAccessNode*);
  virtual void visit(const AcceleratorLiteralNode*);
  virtual void visit(const AcceleratorNegNode*);
  virtual void visit(const AcceleratorSqrtNode*);
  virtual void visit(const AcceleratorAddNode*);
  virtual void visit(const AcceleratorSubNode*);
  virtual void visit(const AcceleratorMulNode*);
  virtual void visit(const AcceleratorDivNode*);
  virtual void visit(const AcceleratorReductionNode*);
  virtual void visit(const AcceleratorDynamicIndex*);

  virtual void visit(const AcceleratorForallNode*);
  virtual void visit(const AcceleratorAssignmentNode*); 

  virtual void visit(const AcceleratorBinaryExprNode*);
  virtual void visit(const AcceleratorUnaryExprNode*);

};

#define ACCEL_RULE(Rule)                                                       \
std::function<void(const Rule*)> Rule##Func;                                   \
std::function<void(const Rule*, AcceleratorMatcher*)> Rule##CtxFunc;           \
void unpack(std::function<void(const Rule*)> pattern) {                        \
  taco_iassert(!Rule##CtxFunc && !Rule##Func);                                 \
  Rule##Func = pattern;                                                        \
}                                                                              \
void unpack(std::function<void(const Rule*, AcceleratorMatcher*)> pattern) {   \
  taco_iassert(!Rule##CtxFunc && !Rule##Func);                                 \
  Rule##CtxFunc = pattern;                                                     \
}                                                                              \
void visit(const Rule* op) {                                                   \
  if (Rule##Func) {                                                            \
    Rule##Func(op);                                                            \
  }                                                                            \
  else if (Rule##CtxFunc) {                                                    \
    Rule##CtxFunc(op, this);                                                   \
    return;                                                                    \
  }                                                                            \
 AcceleratorNotationVisitor::visit(op);                                        \
}

class AcceleratorMatcher : public AcceleratorNotationVisitor {
public:
  template <class AcceleratorExpr>
  void acceleratorMatch(AcceleratorExpr acceleratorExpr) {
    acceleratorExpr.accept(this);
  }

  template <class IR, class... Patterns>
  void process(IR ir, Patterns... patterns) {
    unpack(patterns...);
    ir.accept(this);
  }

private:
  template <class First, class... Rest>
  void unpack(First first, Rest... rest) {
    unpack(first);
    unpack(rest...);
  }

  using AcceleratorNotationVisitor::visit;
  ACCEL_RULE(AcceleratorAccessNode)
  ACCEL_RULE(AcceleratorLiteralNode)
  ACCEL_RULE(AcceleratorNegNode)
  ACCEL_RULE(AcceleratorSqrtNode)
  ACCEL_RULE(AcceleratorAddNode)
  ACCEL_RULE(AcceleratorSubNode)
  ACCEL_RULE(AcceleratorDivNode)
  ACCEL_RULE(AcceleratorMulNode)
  ACCEL_RULE(AcceleratorReductionNode)
  ACCEL_RULE(AcceleratorDynamicIndex)


  ACCEL_RULE(AcceleratorForallNode)
  ACCEL_RULE(AcceleratorAssignmentNode)

};


template <class AcceleratorExpr, class... Patterns>
void acceleratorMatch(AcceleratorExpr acceleratorExpr, Patterns... patterns) {
  if (!acceleratorExpr.defined()) {
    return;
  }
  AcceleratorMatcher().process(acceleratorExpr, patterns...);
}

}

#endif