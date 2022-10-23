#ifndef ACCELERATOR_NOTATION_NODES_ABSTRACT_H
#define ACCELERATOR_NOTATION_NODES_ABSTRACT_H

#include <vector>
#include <memory>

#include "taco/type.h"
#include "taco/util/uncopyable.h"
#include "taco/util/intrusive_ptr.h"

namespace taco {

class TensorVar;
class IndexVar;
class Precompute;
class AcceleratorExprVisitorStrict;
class AcceleratorStmtVisitorStrict;

class DynamicExprVisitorStrict;
class DynamicStmtVisitorStrict;

/// A node of a scalar index expression tree.
struct AcceleratorExprNode : public util::Manageable<AcceleratorExprNode>,
                       private util::Uncopyable {
public:
  AcceleratorExprNode() = default;
  AcceleratorExprNode(Datatype type);
  virtual ~AcceleratorExprNode() = default;

  //TODO: NEED TO DEFINE VISITORS
  virtual void accept(AcceleratorExprVisitorStrict*) const = 0;

  /// Return the scalar data type of the index expression.
  Datatype getDataType() const;

private:
  Datatype dataType;

};


/// A node in a tensor index expression tree
struct AcceleratorStmtNode : public util::Manageable<AcceleratorStmtNode>,
                       private util::Uncopyable {
public:
  AcceleratorStmtNode() = default;
  AcceleratorStmtNode(Type type);
  virtual ~AcceleratorStmtNode() = default;

  //TODO: NEED TO DEFINE VISITORS
  virtual void accept(AcceleratorStmtVisitorStrict*) const = 0;

  Type getType() const;

private:
  Type type;
};

struct DynamicExprNode : public util::Manageable<DynamicExprNode>,
                       private util::Uncopyable {
public:
  DynamicExprNode() = default;
  // DynamicExprNode(Type type);
  virtual ~DynamicExprNode() = default;

  //TODO: NEED TO DEFINE VISITORS
  virtual void accept(DynamicExprVisitorStrict*) const = 0;

private:
  Type type;
};

struct DynamicStmtNode : public util::Manageable<DynamicStmtNode>,
                       private util::Uncopyable {
public:
  DynamicStmtNode() = default;
  // DynamicStmtNode(Type type);
  virtual ~DynamicStmtNode() = default;

  //TODO: NEED TO DEFINE VISITORS
  virtual void accept(DynamicStmtVisitorStrict*) const = 0;

private:
  Type type;
};

}
#endif
