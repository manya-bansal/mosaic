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
struct AcceleratorStmtNode : public util::Manageable<AcceleratorExprNode>,
                       private util::Uncopyable {
public:
  AcceleratorStmtNode() = default;
  AcceleratorStmtNode(Type type);
  virtual ~AcceleratorStmtNode() = default;

  //TODO: NEED TO DEFINE VISITORS

  Type getType() const;

private:
  Type type;
};

}
#endif
