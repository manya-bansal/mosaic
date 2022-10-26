#ifndef TACO_ACCELERATOR_NOTATION_REWRITER_H
#define TACO_ACCELERATOR_NOTATION_REWRITER_H

#include <map>

#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/accelerator_notation/accelerator_notation_visitor.h"

namespace taco {


/// Extend this class to rewrite all index expressions.
class AcceleratorExprRewriterStrict : public AcceleratorExprVisitorStrict {
public:
    virtual ~AcceleratorExprRewriterStrict() {}

     /// Rewrite an index expression.
    AcceleratorExpr rewrite(AcceleratorExpr);

protected:
    /// Assign to expr in visit methods to replace the visited expr.
    AcceleratorExpr expr;

    using AcceleratorExprVisitorStrict::visit;

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


/// Extend this class to rewrite all index statements.
class AcceleratorStmtRewriterStrict : public AcceleratorStmtVisitorStrict {
public:
    virtual ~AcceleratorStmtRewriterStrict() {}

    /// Rewrite an index statement.
    AcceleratorStmt rewrite(AcceleratorStmt);

protected:
    /// Assign to stmt in visit methods to replace the visited stmt.
    AcceleratorStmt stmt;

    using AcceleratorStmtVisitorStrict::visit;

    virtual void visit(const AcceleratorAssignmentNode*) = 0;
    virtual void visit(const AcceleratorForallNode*) = 0; 
};


/// Extend this class to rewrite all index expressions and statements.
class AcceleratorNotationRewriterStrict : public AcceleratorExprRewriterStrict,
                                    public AcceleratorStmtRewriterStrict {
public:
  virtual ~AcceleratorNotationRewriterStrict() {}

  using AcceleratorExprRewriterStrict::rewrite;
  using AcceleratorStmtRewriterStrict::rewrite;

protected:
  using AcceleratorExprRewriterStrict::visit;
  using AcceleratorStmtVisitorStrict::visit;
};


/// Extend this class to rewrite some index expressions and statements.
class AcceleratorNotationRewriter : public AcceleratorNotationRewriterStrict {
public:
  virtual ~AcceleratorNotationRewriter() {}

protected:
  using AcceleratorNotationRewriterStrict::visit;

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
};

}

#endif