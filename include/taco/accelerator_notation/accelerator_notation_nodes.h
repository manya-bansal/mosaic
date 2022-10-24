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

struct AcceleratorDynamicIndexNode : public AcceleratorExprNode {
  AcceleratorDynamicIndexNode(const TensorObject &t, const std::vector<IndexObject> &indexObject) : t(t), indexObject(indexObject) {}
  TensorObject t;
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
  AcceleratorAssignmentNode(const AcceleratorDynamicIndex& dynamicIndex, const AcceleratorExpr& rhs, const AcceleratorExpr& op)
    : dynamicIndex(dynamicIndex), rhs(rhs), op(op) {}

  void accept(AcceleratorStmtVisitorStrict* v) const {
    v->visit(this);
  }

  AcceleratorAccess    lhs;
  AcceleratorDynamicIndex dynamicIndex;
  AcceleratorExpr rhs;
  AcceleratorExpr op;
};

struct DynamicIndexIteratorNode : public DynamicExprNode {
  public:
    DynamicIndexIteratorNode() :  DynamicExprNode() {}
    DynamicIndexIteratorNode(const DynamicOrder& dynamicOrder) : dynamicOrder(dynamicOrder) {}
    void accept(DynamicExprVisitorStrict* v) const override{
      v->visit(this);
    }
    DynamicOrder dynamicOrder;
};

struct DynamicIndexAccessNode : public DynamicExprNode {
  public:
    DynamicIndexAccessNode() :  DynamicExprNode() {}
    DynamicIndexAccessNode(const DynamicOrder& dynamicOrder) : dynamicOrder(dynamicOrder) {}
    void accept(DynamicExprVisitorStrict* v) const override{
      v->visit(this);
    }
    DynamicOrder dynamicOrder;
};

struct DynamicLiteralNode : public DynamicExprNode {
  public:
    DynamicLiteralNode() : DynamicExprNode() {}
    DynamicLiteralNode(const int& num) : num(num) {}
    void accept(DynamicExprVisitorStrict* v) const override{
      v->visit(this);
    }
    int num;
};

struct DynamicIndexLenNode : public DynamicExprNode {
  public:
    DynamicIndexLenNode() : DynamicExprNode() {}
    DynamicIndexLenNode(const DynamicOrder& dynamicOrder) : dynamicOrder(dynamicOrder) {}
    void accept(DynamicExprVisitorStrict* v) const override{
      v->visit(this);
    }
    DynamicOrder dynamicOrder;
};

struct DynamicIndexMulInternalNode : public DynamicExprNode {
  public:
    DynamicIndexMulInternalNode() : DynamicExprNode() {}
    DynamicIndexMulInternalNode(const DynamicOrder& dynamicOrder) : dynamicOrder(dynamicOrder) {}
    void accept(DynamicExprVisitorStrict* v) const override{
      v->visit(this);
    }
    DynamicOrder dynamicOrder;
};

struct DynamicBinaryExprNode : public DynamicExprNode {
  virtual std::string getOperatorString() const = 0;

  DynamicExpr a;
  DynamicExpr b;

protected:
  DynamicBinaryExprNode() : DynamicExprNode() {}
  DynamicBinaryExprNode(DynamicExpr a, DynamicExpr b)
      : DynamicExprNode(), a(a), b(b) {}
};

struct DynamicAddNode : public DynamicBinaryExprNode {
  DynamicAddNode() : DynamicBinaryExprNode() {}
  DynamicAddNode(DynamicExpr a, DynamicExpr b) : DynamicBinaryExprNode(a, b) {}

  std::string getOperatorString() const override{
    return "+";
  }

  void accept(DynamicExprVisitorStrict* v) const override{
    v->visit(this);
  }
};

struct DynamicSubNode : public DynamicBinaryExprNode {
  DynamicSubNode() : DynamicBinaryExprNode() {}
  DynamicSubNode(DynamicExpr a, DynamicExpr b) : DynamicBinaryExprNode(a, b) {}

  std::string getOperatorString() const override{
    return "-";
  }

  void accept(DynamicExprVisitorStrict* v) const override{
    v->visit(this);
  }
};

struct DynamicMulNode : public DynamicBinaryExprNode {
  DynamicMulNode() : DynamicBinaryExprNode() {}
  DynamicMulNode(DynamicExpr a, DynamicExpr b) : DynamicBinaryExprNode(a, b) {}

  std::string getOperatorString() const override{
    return "*";
  }

  void accept(DynamicExprVisitorStrict* v) const override{
    v->visit(this);
  }
};

struct DynamicDivNode : public DynamicBinaryExprNode {
  DynamicDivNode() : DynamicBinaryExprNode() {}
  DynamicDivNode(DynamicExpr a, DynamicExpr b) : DynamicBinaryExprNode(a, b) {}

  std::string getOperatorString() const override{
    return "/";
  }

  void accept(DynamicExprVisitorStrict* v) const override{
    v->visit(this);
  }
};

struct DynamicModNode : public DynamicBinaryExprNode {
  DynamicModNode() : DynamicBinaryExprNode() {}
  DynamicModNode(DynamicExpr a, DynamicExpr b) : DynamicBinaryExprNode(a, b) {}

  std::string getOperatorString() const override{
    return "%";
  }

  void accept(DynamicExprVisitorStrict* v) const override{
    v->visit(this);
  }
};

struct DynamicIndexVarNode : public DynamicExprNode {
  DynamicIndexVarNode() : DynamicExprNode() {}
  DynamicIndexVarNode(IndexVar i) : i(i) {}
  
  IndexVar i; 

  void accept(DynamicExprVisitorStrict* v) const override{
    v->visit(this);
  }
};

struct DynamicEqualNode : public DynamicStmtNode {
  DynamicEqualNode() : DynamicStmtNode() {}
  DynamicEqualNode(DynamicExpr a, DynamicExpr b) : a(a), b(b) {}

  DynamicExpr a;
  DynamicExpr b;

  void accept(DynamicStmtVisitorStrict* v) const override{
    v->visit(this);
  }
};

struct DynamicNotEqualNode : public DynamicStmtNode {
  DynamicNotEqualNode() : DynamicStmtNode() {}
  DynamicNotEqualNode(DynamicExpr a, DynamicExpr b) : a(a), b(b) {}

  DynamicExpr a;
  DynamicExpr b;

  void accept(DynamicStmtVisitorStrict* v) const override{
    v->visit(this);
  }
};

struct DynamicGreaterNode : public DynamicStmtNode {
  DynamicGreaterNode() : DynamicStmtNode() {}
  DynamicGreaterNode(DynamicExpr a, DynamicExpr b) : a(a), b(b) {}

  DynamicExpr a;
  DynamicExpr b;

  void accept(DynamicStmtVisitorStrict* v) const override{
    v->visit(this);
  }
};

struct DynamicLessNode : public DynamicStmtNode {
  DynamicLessNode() : DynamicStmtNode() {}
  DynamicLessNode(DynamicExpr a, DynamicExpr b) : a(a), b(b) {}

  DynamicExpr a;
  DynamicExpr b;

  void accept(DynamicStmtVisitorStrict* v) const override{
    v->visit(this);
  }
};

struct DynamicGeqNode : public DynamicStmtNode {
  DynamicGeqNode() : DynamicStmtNode() {}
  DynamicGeqNode(DynamicExpr a, DynamicExpr b) : a(a), b(b) {}

  DynamicExpr a;
  DynamicExpr b;

  void accept(DynamicStmtVisitorStrict* v) const override{
    v->visit(this);
  }
};

struct DynamicLeqNode : public DynamicStmtNode {
  DynamicLeqNode() : DynamicStmtNode() {}
  DynamicLeqNode(DynamicExpr a, DynamicExpr b) : a(a), b(b) {}

  DynamicExpr a;
  DynamicExpr b;

  void accept(DynamicStmtVisitorStrict* v) const override{
    v->visit(this);
  }
};

struct DynamicForallNode : public DynamicStmtNode {
  DynamicForallNode() : DynamicStmtNode() {}
  DynamicForallNode(DynamicIndexIterator it, DynamicStmt stmt) : it(it), stmt(stmt) {}
  
  DynamicIndexIterator it; 
  DynamicStmt stmt;

  void accept(DynamicStmtVisitorStrict* v) const override{
    v->visit(this);
  }
};

struct DynamicExistsNode : public DynamicStmtNode {
  DynamicExistsNode() : DynamicStmtNode() {}
  DynamicExistsNode(DynamicIndexIterator it, DynamicStmt stmt) : it(it), stmt(stmt) {}
  
  DynamicIndexIterator it; 
  DynamicStmt stmt;

  void accept(DynamicStmtVisitorStrict* v) const override{
    v->visit(this);
  }
};

struct DynamicAndNode : public DynamicStmtNode {
  DynamicAndNode() : DynamicStmtNode() {}
  DynamicAndNode(DynamicStmt a, DynamicStmt b) : a(a), b(b) {}
  
  DynamicStmt a; 
  DynamicStmt b;

  void accept(DynamicStmtVisitorStrict* v) const override{
    v->visit(this);
  }
};

struct DynamicOrNode : public DynamicStmtNode {
  DynamicOrNode() : DynamicStmtNode() {}
  DynamicOrNode(DynamicStmt a, DynamicStmt b) : a(a), b(b) {}
  
  DynamicStmt a; 
  DynamicStmt b;

  void accept(DynamicStmtVisitorStrict* v) const override{
    v->visit(this);
  }
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

template <typename E>
inline bool isa(const DynamicExprNode* e) {
  return e != nullptr && dynamic_cast<const E*>(e) != nullptr;
}

template <typename E>
inline const E* to(const DynamicExprNode* e) {
  taco_iassert(isa<E>(e)) <<
      "Cannot convert " << typeid(e).name() << " to " << typeid(E).name();
  return static_cast<const E*>(e);
}

/// Returns true if statement e is of type S.
template <typename S>
inline bool isa(const DynamicStmtNode* s) {
  return s != nullptr && dynamic_cast<const S*>(s) != nullptr;
}

/// Casts the index statement node s to subtype S.
template <typename SubType>
inline const SubType* to(const DynamicStmtNode* s) {
  taco_iassert(isa<SubType>(s)) <<
      "Cannot convert " << typeid(s).name() << " to " << typeid(SubType).name();
  return static_cast<const SubType*>(s);
}

}
#endif
