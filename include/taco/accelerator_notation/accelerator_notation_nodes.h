#ifndef ACCELERATOR_NOTATION_NODES_H
#define ACCELERATOR_NOTATION_NODES_H

#include <vector>
#include <memory>
#include <functional>
#include <numeric>
#include <functional>

#include "taco/type.h"
#include "taco/util/collections.h"
#include "taco/util/comparable.h"
#include "taco/type.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/accelerator_notation/accelerator_notation_nodes_abstract.h"
#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/accelerator_notation/accelerator_notation_visitor.h"
#include "taco/index_notation/intrinsic.h"
#include "taco/util/strings.h"

namespace taco {

struct AcceleratorAccessNode : public AcceleratorExprNode {
  AcceleratorAccessNode(TensorObject tensorObject, const std::vector<IndexVar>& indices, bool isAccessingStructure)
      : AcceleratorExprNode(isAccessingStructure ? Bool : tensorObject.getType().getDataType()), 
        tensorObject(tensorObject), indexVars(indices), 
        isAccessingStructure(isAccessingStructure) {
  }

  void accept(AcceleratorExprVisitorStrict* v) const override{
    v->visit(this);
  }

  // virtual void setAssignment(const Assignment& assignment) {}

  TensorObject tensorObject;
  std::vector<IndexVar> indexVars;
  bool isAccessingStructure;

protected:
  /// Initialize an AccessNode with just a TensorVar. If this constructor is used,
  /// then indexVars must be set afterwards.
  explicit AcceleratorAccessNode(TensorObject tensorObject) : 
      AcceleratorExprNode((tensorObject.getType().getDataType())), 
      tensorObject(tensorObject), isAccessingStructure(false) {}
};

struct AcceleratorLiteralNode : public AcceleratorExprNode {
  template <typename T> 
  explicit AcceleratorLiteralNode(T val) : AcceleratorExprNode(type<T>()) {
    this->val = malloc(sizeof(T));
    *static_cast<T*>(this->val) = val;
  }

  ~AcceleratorLiteralNode() {
    free(val);
  }

  void accept(AcceleratorExprVisitorStrict* v) const override{
    v->visit(this);
  }

  template <typename T> T getVal() const {
    taco_iassert(getDataType() == type<T>())
        << "Attempting to get data of wrong type";
    return *static_cast<T*>(val);
  }

  void* val;
};

struct AcceleratorUnaryExprNode : public AcceleratorExprNode {
  AcceleratorExpr a;

protected:
  explicit AcceleratorUnaryExprNode(AcceleratorExpr a) : AcceleratorExprNode(a.getDataType()), a(a) {}
};


struct AcceleratorNegNode : public AcceleratorUnaryExprNode {
  explicit AcceleratorNegNode(AcceleratorExpr operand) : AcceleratorUnaryExprNode(operand) {}

  void accept(AcceleratorExprVisitorStrict* v) const override{
    v->visit(this);
  }
};


struct AcceleratorBinaryExprNode : public AcceleratorExprNode {
  virtual std::string getOperatorString() const = 0;

  AcceleratorExpr a;
  AcceleratorExpr b;

protected:
  AcceleratorBinaryExprNode() : AcceleratorExprNode() {}
  AcceleratorBinaryExprNode(AcceleratorExpr a, AcceleratorExpr b)
      : AcceleratorExprNode(max_type(a.getDataType(), b.getDataType())), a(a), b(b) {}
};


struct AcceleratorAddNode : public AcceleratorBinaryExprNode {
  AcceleratorAddNode() : AcceleratorBinaryExprNode() {}
  AcceleratorAddNode(AcceleratorExpr a, AcceleratorExpr b) : AcceleratorBinaryExprNode(a, b) {}

  std::string getOperatorString() const override{
    return "+";
  }

  void accept(AcceleratorExprVisitorStrict* v) const override{
    v->visit(this);
  }
};


struct AcceleratorSubNode : public AcceleratorBinaryExprNode {
  AcceleratorSubNode() : AcceleratorBinaryExprNode() {}
  AcceleratorSubNode(AcceleratorExpr a, AcceleratorExpr b) : AcceleratorBinaryExprNode(a, b) {}

  std::string getOperatorString() const override{
    return "-";
  }

  void accept(AcceleratorExprVisitorStrict* v) const override{
    v->visit(this);
  }
};

struct AcceleratorDynamicIndex : public AcceleratorExprNode {
  AcceleratorDynamicIndex(const std::vector<IndexObject> &indexObject) : indexObject(indexObject) {}
  std::vector<IndexObject> indexObject;

  void accept(AcceleratorExprVisitorStrict* v) const override{
    v->visit(this);
  }
};


struct AcceleratorMulNode : public AcceleratorBinaryExprNode {
  AcceleratorMulNode() : AcceleratorBinaryExprNode() {}
  AcceleratorMulNode(AcceleratorExpr a, AcceleratorExpr b) : AcceleratorBinaryExprNode(a, b) {}

  std::string getOperatorString() const {
    return "*";
  }

  void accept(AcceleratorExprVisitorStrict* v) const {
    v->visit(this);
  }
};


struct AcceleratorDivNode : public AcceleratorBinaryExprNode {
  AcceleratorDivNode() : AcceleratorBinaryExprNode() {}
  AcceleratorDivNode(AcceleratorExpr a, AcceleratorExpr b) : AcceleratorBinaryExprNode(a, b) {}

  std::string getOperatorString() const {
    return "/";
  }

  void accept(AcceleratorExprVisitorStrict* v) const {
    v->visit(this);
  }
};


struct AcceleratorSqrtNode : public AcceleratorUnaryExprNode {
  AcceleratorSqrtNode(AcceleratorExpr operand) : AcceleratorUnaryExprNode(operand) {}

  void accept(AcceleratorExprVisitorStrict* v) const {
    v->visit(this);
  }

};

struct AcceleratorReductionNode : public AcceleratorExprNode {
  AcceleratorReductionNode(AcceleratorExpr op, IndexVar var, AcceleratorExpr a) : op(op), var(var), a(a) {}

  void accept(AcceleratorExprVisitorStrict* v) const {
     v->visit(this);
  }

  AcceleratorExpr op;  // The binary reduction operator, which is a `BinaryExprNode`
                      // with undefined operands)
  IndexVar var;
  AcceleratorExpr a;
};


struct AcceleratorForallNode : public AcceleratorStmtNode {
  AcceleratorForallNode(IndexVar indexVar, AcceleratorStmt stmt)
      : indexVar(indexVar), stmt(stmt) {}

  void accept(AcceleratorStmtVisitorStrict* v) const {
    v->visit(this);
  }

  IndexVar indexVar;
  AcceleratorStmt stmt;
};

// Accelerate Statements
struct AcceleratorAssignmentNode : public AcceleratorStmtNode {
  AcceleratorAssignmentNode(const AcceleratorAccess& lhs, const AcceleratorExpr& rhs, const AcceleratorExpr& op)
      : lhs(lhs), rhs(rhs), op(op) {}

  void accept(AcceleratorStmtVisitorStrict* v) const {
    v->visit(this);
  }

  AcceleratorAccess    lhs;
  AcceleratorExpr rhs;
  AcceleratorExpr op;
};


/// Returns true if expression e is of type E.
template <typename E>
inline bool isa(const AcceleratorExprNode* e) {
  return e != nullptr && dynamic_cast<const E*>(e) != nullptr;
}

/// Casts the expression e to type E.
template <typename E>
inline const E* to(const AcceleratorExprNode* e) {
  taco_iassert(isa<E>(e)) <<
      "Cannot convert " << typeid(e).name() << " to " << typeid(E).name();
  return static_cast<const E*>(e);
}

/// Returns true if statement e is of type S.
template <typename S>
inline bool isa(const AcceleratorStmtNode* s) {
  return s != nullptr && dynamic_cast<const S*>(s) != nullptr;
}

/// Casts the index statement node s to subtype S.
template <typename SubType>
inline const SubType* to(const AcceleratorStmtNode* s) {
  taco_iassert(isa<SubType>(s)) <<
      "Cannot convert " << typeid(s).name() << " to " << typeid(SubType).name();
  return static_cast<const SubType*>(s);
}

}
#endif
