#ifndef ACCELERATOR_NOTATION_VISITOR_H
#define ACCELERATOR_NOTATION_VISITOR_H

#include <vector>
#include <functional>
#include "taco/error.h"

namespace taco {

class AcceleratorExpr;

struct AcceleratorAccessNode;
struct AcceleratorLiteralNode;

class AcceleratorExprVisitorStrict {
    public:
        virtual ~AcceleratorExprVisitorStrict() = default;

        void visit(const AcceleratorExpr&);

        virtual void visit(const AcceleratorAccessNode*) = 0;
        virtual void visit(const AcceleratorLiteralNode*) = 0;

};

/// Visit nodes in index notation
class AcceleratorNotationVisitorStrict : public AcceleratorExprVisitorStrict {
    public:
        virtual ~AcceleratorNotationVisitorStrict() = default;

        using AcceleratorExprVisitorStrict::visit;
        // using IndexStmtVisitorStrict::visit;
};

class AcceleratorNotationVisitor : public AcceleratorNotationVisitorStrict {
public:
  virtual ~AcceleratorNotationVisitor() = default;

  using AcceleratorExprVisitorStrict::visit;

  virtual void visit(const AcceleratorAccessNode*);

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