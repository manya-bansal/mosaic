#ifndef ACCELERATOR_NOTATION_H
#define ACCELERATOR_NOTATION_H

#include <functional>
#include <ostream>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <map>
#include <utility>
#include <functional>

#include "taco/util/name_generator.h"
#include "taco/format.h"
#include "taco/error.h"
#include "taco/util/intrusive_ptr.h"
#include "taco/util/comparable.h"
#include "taco/type.h"
#include "taco/ir/ir.h"
#include "taco/codegen/module.h"
#include "taco/index_notation/intrinsic.h"
#include "taco/accelerator_notation/accelerator_notation_nodes_abstract.h"
#include "taco/ir_tags.h"
#include "taco/index_notation/provenance_graph.h"
#include "taco/index_notation/properties.h"
#include "taco/index_notation/index_notation.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"

namespace taco {

class AcceleratorExprVisitorStrict;
class AcceleratorStmtVisitorStrict;
class AcceleratorAssignment;
class TensorObject;

struct AcceleratorAccessNode;
struct AcceleratorLiteralNode;

struct AcceleratorAssignmentNode;

class AcceleratorExpr : public util::IntrusivePtr<const AcceleratorExprNode> {
public:
  AcceleratorExpr() : util::IntrusivePtr<const AcceleratorExprNode>(nullptr) {}
  AcceleratorExpr(const AcceleratorExprNode* n) : util::IntrusivePtr<const AcceleratorExprNode>(n) {}

  /// Construct a scalar tensor access.
  AcceleratorExpr(TensorObject tensorObject);

  /// Consturct an integer literal.
  AcceleratorExpr(char);
  AcceleratorExpr(int8_t);
  AcceleratorExpr(int16_t);
  AcceleratorExpr(int32_t);
  AcceleratorExpr(int64_t);

  /// Consturct an unsigned integer literal.
  AcceleratorExpr(uint8_t);
  AcceleratorExpr(uint16_t);
  AcceleratorExpr(uint32_t);
  AcceleratorExpr(uint64_t);

  /// Consturct double literal.
  AcceleratorExpr(float);
  AcceleratorExpr(double);

  /// Construct complex literal.
  AcceleratorExpr(std::complex<float>);
  AcceleratorExpr(std::complex<double>);

  void accept(AcceleratorExprVisitorStrict *) const;

  Datatype getDataType() const;

  friend std::ostream& operator<<(std::ostream&, const AcceleratorExpr&);

};

class AcceleratorStmt : public util::IntrusivePtr<const AcceleratorStmtNode> {
public:
  AcceleratorStmt();
  AcceleratorStmt(const AcceleratorStmtNode* n) : util::IntrusivePtr<const AcceleratorStmtNode>(n) {}

  void accept(AcceleratorStmtVisitorStrict *) const;

  friend std::ostream& operator<<(std::ostream&, const AcceleratorStmt&);

};

class AcceleratorAccess : public AcceleratorExpr {
public:
    AcceleratorAccess() = default;
    AcceleratorAccess(const AcceleratorAccess&) = default;
    AcceleratorAccess(const AcceleratorAccessNode*);
    AcceleratorAccess(const TensorObject& tensorObject, const std::vector<IndexVar>& indices={},
            bool isAccessingStructure=false);

    /// Assign the result of an expression to a left-hand-side tensor access.
    /// ```
    /// a(i) = b(i) * c(i);
    /// ```
    AcceleratorAssignment operator=(const AcceleratorExpr&);

    /// Must override the default Access operator=, otherwise it is a copy.
    AcceleratorAssignment operator=(const AcceleratorAccess&);

    /// Must disambiguate TensorVar as it can be implicitly converted to IndexExpr
    /// and AccesExpr.
    AcceleratorAssignment operator=(const TensorObject&);

    typedef AcceleratorAccess Node;
};

/// A literal index expression is a scalar literal that is embedded in the code.
/// @note In the future we may allow general tensor literals.
class AcceleratorLiteral : public AcceleratorExpr {
public:
  AcceleratorLiteral() = default;
  explicit AcceleratorLiteral(const AcceleratorLiteralNode*);
 
  explicit AcceleratorLiteral(bool);
  explicit AcceleratorLiteral(unsigned char);
  explicit AcceleratorLiteral(unsigned short);
  explicit AcceleratorLiteral(unsigned int);
  explicit AcceleratorLiteral(unsigned long);
  explicit AcceleratorLiteral(unsigned long long);
  explicit AcceleratorLiteral(char);
  explicit AcceleratorLiteral(short);
  explicit AcceleratorLiteral(int);
  explicit AcceleratorLiteral(long);
  explicit AcceleratorLiteral(long long);
  explicit AcceleratorLiteral(int8_t);
  explicit AcceleratorLiteral(float);
  explicit AcceleratorLiteral(double);
  explicit AcceleratorLiteral(std::complex<float>);
  explicit AcceleratorLiteral(std::complex<double>);

  static AcceleratorLiteral zero(Datatype);

  /// Returns the literal value.
  template <typename T> T getVal() const;

  /// Returns an untyped pointer to the literal value
  void* getValPtr();

  typedef AcceleratorLiteralNode Node;

};

/// An assignment statement assigns an index expression to the locations in a
/// tensor given by an lhs access expression.
class AcceleratorAssignment : public AcceleratorStmt {
public:
  AcceleratorAssignment() = default;
  AcceleratorAssignment(const AcceleratorAssignmentNode*);

  /// Create an assignment. Can specify an optional operator `op` that turns the
  /// assignment into a compound assignment, e.g. `+=`.
  AcceleratorAssignment(AcceleratorAccess lhs, AcceleratorExpr rhs, AcceleratorExpr op = AcceleratorExpr());

  /// Create an assignment. Can specify an optional operator `op` that turns the
  /// assignment into a compound assignment, e.g. `+=`. Additionally, specify
  /// any modifers on reduction index variables (windows, index sets, etc.).
  AcceleratorAssignment(TensorObject tensor, std::vector<IndexVar> indices, AcceleratorExpr rhs,
             AcceleratorExpr op = AcceleratorExpr());

  /// Return the assignment's left-hand side.
  AcceleratorAccess getLhs() const;

  /// Return the assignment's right-hand side.
  AcceleratorExpr getRhs() const;

  /// Return the assignment compound operator (e.g., `+=`) or an undefined
  /// expression if the assignment is not compound (`=`).
  AcceleratorExpr getOperator() const;

  typedef AcceleratorAssignmentNode Node;
};


class TensorObject : public util::Comparable<TensorObject> {
public:
  TensorObject();
  TensorObject(const Type& type);
  TensorObject(const std::string& name, const Type& type);
  TensorObject(const std::string& name, const Type& type, const Format& format);

  /// Returns the name of the tensor object.
  std::string getName() const;

  /// Sets the name of the tensor object.
  void setName(const std::string& name) const;

  /// Returns the type of the tensor object.
  const Type& getType() const;

  /// Returns the order of the tensor object.
  int getOrder() const;

  const AcceleratorAccess operator()(const std::vector<IndexVar>& indices) const;
  AcceleratorAccess operator()(const std::vector<IndexVar>& indices);

  template <typename... IndexVars>
  const AcceleratorAccess operator()(const IndexVars&... indices) const {
    return static_cast<const TensorObject*>(this)->operator()({indices...});
  }

  /// Create an index expression that accesses (reads or writes) this tensor.
  template <typename... IndexVars>
  AcceleratorAccess operator()(const IndexVars&... indices) {
    return this->operator()({indices...});
  }
  
  /// Assign a scalar expression to a scalar tensor.
  AcceleratorAssignment operator=(AcceleratorExpr);

  // /// Add a scalar expression to a scalar tensor.
  // AcceleratorAssignment operator+=(AcceleratorExpr);


private:
  struct Content;
  std::shared_ptr<Content> content;

};

std::ostream& operator<<(std::ostream&, const TensorObject&);

}

#endif 