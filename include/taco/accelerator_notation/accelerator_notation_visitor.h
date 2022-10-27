#ifndef ACCELERATOR_NOTATION_VISITOR_H
#define ACCELERATOR_NOTATION_VISITOR_H

#include <vector>
#include <functional>
#include "taco/error.h"

namespace taco {

class AcceleratorExpr;
class AcceleratorStmt;

class DynamicExpr;
class DynamicStmt;

class PropertyExpr;
class PropertyStmt;

struct AcceleratorAccessNode;
struct AcceleratorLiteralNode;
struct AcceleratorNegNode;
struct AcceleratorSqrtNode;
struct AcceleratorAddNode;
struct AcceleratorSubNode;
struct AcceleratorMulNode;
struct AcceleratorDivNode;
struct AcceleratorReductionNode;
struct AcceleratorDynamicIndexNode;

struct AcceleratorBinaryExprNode;
struct AcceleratorUnaryExprNode;

struct AcceleratorAssignmentNode;
struct AcceleratorForallNode;

struct DynamicIndexIteratorNode;
struct DynamicIndexAccessNode;
struct DynamicLiteralNode;
struct DynamicIndexLenNode;
struct DynamicIndexMulInternalNode;
struct DynamicAddNode;
struct DynamicSubNode;
struct DynamicMulNode;
struct DynamicDivNode;
struct DynamicModNode;

struct DynamicEqualNode;
struct DynamicNotEqualNode;
struct DynamicGreaterNode;
struct DynamicLessNode;
struct DynamicGeqNode;
struct DynamicLeqNode;
struct DynamicForallNode;
struct DynamicExistsNode;
struct DynamicIndexVarNode;
struct DynamicAndNode;
struct DynamicOrNode;

struct PropertyTagNode;
struct PropertyAddNode;
struct PropertySubNode;
struct PropertyMulNode;
struct PropertyDivNode;
struct PropertyAssignNode;

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
        virtual void visit(const AcceleratorDynamicIndexNode*) = 0;

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
  virtual void visit(const AcceleratorDynamicIndexNode*);

  virtual void visit(const AcceleratorForallNode*);
  virtual void visit(const AcceleratorAssignmentNode*); 

  virtual void visit(const AcceleratorBinaryExprNode*);
  virtual void visit(const AcceleratorUnaryExprNode*);

};

class DynamicExprVisitorStrict {
    public:
        virtual ~DynamicExprVisitorStrict() = default;

        void visit(const DynamicExpr&);

        virtual void visit(const DynamicIndexIteratorNode*) = 0;
        virtual void visit(const DynamicIndexAccessNode*) = 0;
        virtual void visit(const DynamicLiteralNode*) = 0;
        virtual void visit(const DynamicIndexLenNode*) = 0;
        virtual void visit(const DynamicIndexMulInternalNode*) = 0;
        virtual void visit(const DynamicAddNode*) = 0;
        virtual void visit(const DynamicSubNode*) = 0;
        virtual void visit(const DynamicMulNode*) = 0;
        virtual void visit(const DynamicDivNode*) = 0;
        virtual void visit(const DynamicModNode*) = 0;
        virtual void visit(const DynamicIndexVarNode*) = 0;

};

class DynamicStmtVisitorStrict {
    public:
        virtual ~DynamicStmtVisitorStrict() = default;

        void visit(const DynamicStmt&);

        virtual void visit(const DynamicEqualNode*) = 0;
        virtual void visit(const DynamicNotEqualNode*) = 0;
        virtual void visit(const DynamicGreaterNode*) = 0;
        virtual void visit(const DynamicLessNode*) = 0;
        virtual void visit(const DynamicGeqNode*) = 0;
        virtual void visit(const DynamicLeqNode*) = 0;
        virtual void visit(const DynamicForallNode*) = 0;
        virtual void visit(const DynamicExistsNode*) = 0;
        virtual void visit(const DynamicAndNode*) = 0; 
        virtual void visit(const DynamicOrNode*) = 0; 
};

class DynamicNotationVisitorStrict : public DynamicExprVisitorStrict,
                                     public DynamicStmtVisitorStrict{
    public:
        virtual ~DynamicNotationVisitorStrict() = default;

        using DynamicExprVisitorStrict::visit;
        using DynamicStmtVisitorStrict::visit;
};

class DynamicNotationVisitor : public DynamicNotationVisitorStrict {
public:
  virtual ~DynamicNotationVisitor() = default;

  using DynamicNotationVisitorStrict::visit;

  virtual void visit(const DynamicIndexIteratorNode*);
  virtual void visit(const DynamicIndexAccessNode*);
  virtual void visit(const DynamicLiteralNode*);
  virtual void visit(const DynamicIndexLenNode*);
  virtual void visit(const DynamicIndexMulInternalNode*);
  virtual void visit(const DynamicAddNode*);
  virtual void visit(const DynamicSubNode*);
  virtual void visit(const DynamicMulNode*);
  virtual void visit(const DynamicDivNode*);
  virtual void visit(const DynamicModNode*);
  virtual void visit(const DynamicIndexVarNode*);
  virtual void visit(const DynamicEqualNode*);
  virtual void visit(const DynamicNotEqualNode*);
  virtual void visit(const DynamicGreaterNode*);
  virtual void visit(const DynamicLessNode*);
  virtual void visit(const DynamicLeqNode*);
  virtual void visit(const DynamicGeqNode*);
  virtual void visit(const DynamicForallNode*);
  virtual void visit(const DynamicExistsNode*);
  virtual void visit(const DynamicAndNode*);
  virtual void visit(const DynamicOrNode*); 

};

class PropertyExprVisitorStrict {
  public:
    virtual ~PropertyExprVisitorStrict() = default;

    void visit(const PropertyExpr&);

    virtual void visit(const PropertyTagNode*) = 0;
    virtual void visit(const PropertyAddNode*) = 0;
    virtual void visit(const PropertySubNode*) = 0;
    virtual void visit(const PropertyMulNode*) = 0;
    virtual void visit(const PropertyDivNode*) = 0;
};

class PropertyStmtVisitorStrict {
    public:
      virtual ~PropertyStmtVisitorStrict() = default;

      void visit(const PropertyStmt&);

      virtual void visit(const PropertyAssignNode*) = 0;
};

class PropertyNotationVisitorStrict : public PropertyExprVisitorStrict,
                                     public PropertyStmtVisitorStrict{
    public:
        virtual ~PropertyNotationVisitorStrict() = default;

        using PropertyExprVisitorStrict::visit;
        using PropertyStmtVisitorStrict::visit;
};

class PropertyNotationVisitor : public DynamicNotationVisitorStrict {
public:
  virtual ~PropertyNotationVisitor() = default;
  
  using DynamicNotationVisitorStrict::visit;

  virtual void visit(const PropertyTagNode*);
  virtual void visit(const PropertyAddNode*);
  virtual void visit(const PropertyMulNode*);
  virtual void visit(const PropertyDivNode*);
  virtual void visit(const PropertySubNode*);
  virtual void visit(const PropertyAssignNode*);

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
  ACCEL_RULE(AcceleratorDynamicIndexNode)


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