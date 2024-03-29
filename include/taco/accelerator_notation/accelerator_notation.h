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
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_notation/accelerator_notation_nodes_abstract.h"
#include "taco/ir_tags.h"
#include "taco/index_notation/provenance_graph.h"
#include "taco/index_notation/properties.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"

namespace taco {

class AcceleratorExprVisitorStrict;
class AcceleratorStmtVisitorStrict;

class AcceleratorAssignment;
class TensorObject;

struct AcceleratorAccessNode;
struct AcceleratorLiteralNode;
struct AcceleratorNegNode;
struct AcceleratorAddNode;
struct AcceleratorSubNode;
struct AcceleratorMulNode;
struct AcceleratorDivNode;
struct AcceleratorSqrtNode;
struct AcceleratorReductionNode;
struct AcceleratorDynamicIndexNode;

struct AcceleratorForallNode;
struct AcceleratorAssignmentNode;

struct DynamicLiteralNode;
struct DynamicIndexIteratorNode;
struct DynamicIndexMulInternalNode;
struct DynamicIndexAccessNode;
struct DynamicIndexLenNode;
struct DynamicAddNode;
struct DynamicMulNode;
struct DynamicDivNode;
struct DynamicSubNode;
struct DynamicModNode;
struct DynamicIndexVarNode;

struct DynamicEqualNode;
struct DynamicNotEqualNode;
struct DynamicGreaterNode;
struct DynamicLessNode;
struct DynamicGeqNode;
struct DynamicLeqNode;
struct DynamicForallNode;
struct DynamicExistsNode;
struct DynamicAndNode;
struct DynamicOrNode;

class DynamicStmtVisitorStrict;

struct PropertyTagNode;
struct PropertyAddNode;
struct PropertyDivNode;
struct PropertyMulNode;
struct PropertySubNode;
struct PropertyAssignNode;

struct PropertyAssign;

class IndexVar;


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

  std::vector<IndexVar> getIndexVars() const;

  std::map<IndexVar,Dimension> getIndexVarDomains() const;

  friend std::ostream& operator<<(std::ostream&, const AcceleratorExpr&);

};

/// Return true if the index statement is of the given subtype.  The subtypes
/// are Assignment, Forall, Where, Sequence, and Multi.
template <typename SubType> bool isa(AcceleratorExpr);

/// Casts the index statement to the given subtype. Assumes S is a subtype and
/// the subtypes are Assignment, Forall, Where, Sequence, and Multi.
template <typename SubType> SubType to(AcceleratorExpr);

/// Construct and returns an expression that negates this expression.
/// ```
/// A(i,j) = -B(i,j);
/// ```
AcceleratorExpr operator-(const AcceleratorExpr&);

/// Add two index expressions.
/// ```
/// A(i,j) = B(i,j) + C(i,j);
/// ```
AcceleratorExpr operator+(const AcceleratorExpr&, const AcceleratorExpr&);

/// Subtract two index expressions.
/// ```
/// A(i,j) = B(i,j) - C(i,j);
/// ```
AcceleratorExpr operator-(const AcceleratorExpr&, const AcceleratorExpr&);

/// Multiply two index expressions.
/// ```
/// A(i,j) = B(i,j) * C(i,j);  // Component-wise multiplication
/// ```
AcceleratorExpr operator*(const AcceleratorExpr&, const AcceleratorExpr&);

/// Divide an index expression by another.
/// ```
/// A(i,j) = B(i,j) / C(i,j);  // Component-wise division
/// ```
AcceleratorExpr operator/(const AcceleratorExpr&, const AcceleratorExpr&);



class AcceleratorStmt : public util::IntrusivePtr<const AcceleratorStmtNode> {
public:
  AcceleratorStmt() : util::IntrusivePtr<const AcceleratorStmtNode>(nullptr) {}
  AcceleratorStmt(const AcceleratorStmtNode* n) : util::IntrusivePtr<const AcceleratorStmtNode>(n) {}

  void accept(AcceleratorStmtVisitorStrict *) const;

  std::vector<IndexVar> getIndexVars() const;

  friend std::ostream& operator<<(std::ostream&, const AcceleratorStmt&);

};

template <typename SubType> bool isa(AcceleratorStmt);

/// Casts the index statement to the given subtype. Assumes S is a subtype and
/// the subtypes are Assignment, Forall, Where, Multi, and Sequence.
template <typename SubType> SubType to(AcceleratorStmt);

class AcceleratorAccess : public AcceleratorExpr {
public:
    AcceleratorAccess() = default;
    AcceleratorAccess(const AcceleratorAccess&) = default;
    AcceleratorAccess(const AcceleratorAccessNode*);
    AcceleratorAccess(const TensorObject& tensorObject, const std::vector<IndexVar>& indices={},
            bool isAccessingStructure=false);

    const TensorObject& getTensorObject() const;
    const std::vector<IndexVar>& getIndexVars() const;

    /// Assign the result of an expression to a left-hand-side tensor access.
    /// ```
    /// a(i) = b(i) * c(i);
    /// ```
    AcceleratorAssignment operator=(const AcceleratorExpr&);

    AcceleratorAssignment operator=(const AcceleratorExpr&) const;

    /// Must override the default Access operator=, otherwise it is a copy.
    AcceleratorAssignment operator=(const AcceleratorAccess&);

    AcceleratorAssignment operator=(const AcceleratorAccess&) const;

    /// Must disambiguate TensorVar as it can be implicitly converted to IndexExpr
    /// and AccesExpr.
    AcceleratorAssignment operator=(const TensorObject&);

    AcceleratorAssignment operator+=(const AcceleratorExpr& expr);

    typedef AcceleratorAccessNode Node;
};

class AcceleratorDynamicIndex : public AcceleratorExpr {
public:
    AcceleratorDynamicIndex() = default;
    AcceleratorDynamicIndex(const AcceleratorDynamicIndex&) = default;
    AcceleratorDynamicIndex(const AcceleratorDynamicIndexNode*);
    AcceleratorDynamicIndex(const TensorObject& tensorObject, const std::vector<IndexObject>& indices={});

    const TensorObject& getTensorObject() const;
    const std::vector<IndexObject>& getIndexObjects() const;

    /// Assign the result of an expression to a left-hand-side tensor access.
    /// ```
    /// a(i) = b(i) * c(i);
    /// ```
    AcceleratorAssignment operator=(const AcceleratorExpr&);

    AcceleratorAssignment operator=(const AcceleratorExpr&) const;

    /// Must override the default Access operator=, otherwise it is a copy.
    AcceleratorAssignment operator=(const AcceleratorAccess&);

    AcceleratorAssignment operator=(const AcceleratorAccess&) const;

    AcceleratorAssignment operator=(const AcceleratorDynamicIndex&);

    AcceleratorAssignment operator=(const AcceleratorDynamicIndex&) const;

    /// Must disambiguate TensorVar as it can be implicitly converted to IndexExpr
    /// and AccesExpr.
    AcceleratorAssignment operator=(const TensorObject&);

    AcceleratorAssignment operator+=(const AcceleratorExpr& expr);

    typedef AcceleratorDynamicIndexNode Node;
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

/// A neg expression computes negates a number.
/// ```
/// a(i) = -b(i);
/// ```
class AcceleratorNeg : public AcceleratorExpr {
public:
  AcceleratorNeg() = default;
  AcceleratorNeg(const AcceleratorNegNode*);
  AcceleratorNeg(AcceleratorExpr a);

  AcceleratorExpr getA() const;

  typedef AcceleratorNegNode Node;
};

/// An add expression adds two numbers.
/// ```
/// a(i) = b(i) + c(i);
/// ```
class AcceleratorAdd : public AcceleratorExpr {
public:
  AcceleratorAdd();
  AcceleratorAdd(const AcceleratorAddNode*);
  AcceleratorAdd(AcceleratorExpr a, AcceleratorExpr b);

  AcceleratorExpr getA() const;
  AcceleratorExpr getB() const;

  typedef AcceleratorAddNode Node;
};

/// A sub expression subtracts two numbers.
/// ```
/// a(i) = b(i) - c(i);
/// ```
class AcceleratorSub : public AcceleratorExpr {
public:
  AcceleratorSub();
  AcceleratorSub(const AcceleratorSubNode*);
  AcceleratorSub(AcceleratorExpr a, AcceleratorExpr b);

  AcceleratorExpr getA() const;
  AcceleratorExpr getB() const;

  typedef AcceleratorSubNode Node;
};

/// An mull expression multiplies two numbers.
/// ```
/// a(i) = b(i) * c(i);
/// ```
class AcceleratorMul : public AcceleratorExpr {
public:
  AcceleratorMul();
  AcceleratorMul(const AcceleratorMulNode*);
  AcceleratorMul(AcceleratorExpr a, AcceleratorExpr b);

  AcceleratorExpr getA() const;
  AcceleratorExpr getB() const;

  typedef AcceleratorMulNode Node;
};


/// An div expression divides two numbers.
/// ```
/// a(i) = b(i) / c(i);
/// ```
class AcceleratorDiv : public AcceleratorExpr {
public:
  AcceleratorDiv();
  AcceleratorDiv(const AcceleratorDivNode*);
  AcceleratorDiv(AcceleratorExpr a, AcceleratorExpr b);

  AcceleratorExpr getA() const;
  AcceleratorExpr getB() const;

  typedef AcceleratorDivNode Node;
};


/// A sqrt expression computes the square root of a number
/// ```
/// a(i) = sqrt(b(i));
/// ```
class AcceleratorSqrt : public AcceleratorExpr {
public:
  AcceleratorSqrt() = default;
  AcceleratorSqrt(const AcceleratorSqrtNode*);
  AcceleratorSqrt(AcceleratorExpr a);

  AcceleratorExpr getA() const;

  typedef AcceleratorSqrtNode Node;
};

/// A reduction over the components AcceleratorExpr by the reduction variable.
class AcceleratorReduction : public AcceleratorExpr {
public:
  AcceleratorReduction() = default;
  AcceleratorReduction(const AcceleratorReductionNode*);
  AcceleratorReduction(AcceleratorExpr op, IndexVar var, AcceleratorExpr expr);

  AcceleratorExpr getOp() const;
  IndexVar getVar() const;
  AcceleratorExpr getExpr() const;

  typedef AcceleratorReductionNode Node;
};

/// Create a summation index expression.
AcceleratorReduction sum(IndexVar i, AcceleratorExpr expr);


/// A forall statement binds an index variable to values and evaluates the
/// sub-statement for each of these values.
class AcceleratorForall : public AcceleratorStmt {
public:
  AcceleratorForall() = default;
  AcceleratorForall(const AcceleratorForallNode*);
  AcceleratorForall(IndexVar indexVar, AcceleratorStmt stmt);

  IndexVar getIndexVar() const;
  AcceleratorStmt getStmt() const;

  typedef AcceleratorForallNode Node;
};

/// Create a AcceleratorForall index statement.
AcceleratorForall forall(IndexVar i, AcceleratorStmt stmt);


/// An assignment statement assigns an index expression to the locations in a
/// tensor given by an lhs access expression.
class AcceleratorAssignment : public AcceleratorStmt {
public:
  AcceleratorAssignment() = default;
  AcceleratorAssignment(const AcceleratorAssignmentNode*);

  /// Create an assignment. Can specify an optional operator `op` that turns the
  /// assignment into a compound assignment, e.g. `+=`.
  AcceleratorAssignment(AcceleratorAccess lhs, AcceleratorExpr rhs, AcceleratorExpr op = AcceleratorExpr());

  AcceleratorAssignment(AcceleratorDynamicIndex lhs, AcceleratorExpr rhs, AcceleratorExpr op = AcceleratorExpr());

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

  /// Return the free index variables in the assignment, which are those used to
  /// access the left-hand side.
  const std::vector<IndexVar>& getFreeVars() const;

  /// get reduction vars 
  const std::vector<IndexVar> getImplicitReducionVars() const;

  typedef AcceleratorAssignmentNode Node;
};

AcceleratorAssignment makeReductionNotation(AcceleratorAssignment assignment);


class TensorObject : public util::Comparable<TensorObject> {
public:
  TensorObject();
  TensorObject(const Type& type);
  TensorObject(const std::string& name, const Type& type);
  TensorObject(const Type& type, const Format& format);
  TensorObject(const std::string& name, const Type& type, const Format& format);

  /// Returns the name of the tensor object.
  std::string getName() const;

  /// Sets the name of the tensor object.
  void setName(const std::string& name) const;

  /// Returns the type of the tensor object.
  const Type& getType() const;

  const Format& getFormat() const;

  /// Returns the order of the tensor object.
  int getOrder() const;

  std::set<std::string> getProperties() const;

  void addProperty(std::string property) const;

  bool hasProperty(std::string property) const;

  const AcceleratorAccess operator()(const std::vector<IndexVar>& indices) const;
 
  template <typename... IndexVars>
  const AcceleratorAccess operator()(const IndexVars&... indices) const {
    return static_cast<const TensorObject*>(this)->operator()({indices...});
  }

  AcceleratorAccess operator()(const std::vector<IndexVar>& indices);
  /// Create an index expression that accesses (reads or writes) this tensor.
  template <typename... IndexVars>
  AcceleratorAccess operator()(const IndexVars&... indices) {
    return this->operator()({indices...});
  }

  AcceleratorDynamicIndex operator[](const std::vector<IndexObject>& indices) const;
  
  /// Assign a scalar expression to a scalar tensor.
  AcceleratorAssignment operator=(AcceleratorExpr);

  AcceleratorAssignment operator=(AcceleratorExpr) const;

  // AcceleratorAssignment operator=(const AcceleratorAccess& access);
  // AcceleratorAssignment operator=(const AcceleratorAccess& access) const;

  // AcceleratorAssignment operator=(const AcceleratorDynamicIndex& access);
  // AcceleratorAssignment operator=(const AcceleratorDynamicIndex& access) const;

  // /// Add a scalar expression to a scalar tensor.
  AcceleratorAssignment operator+=(AcceleratorExpr);

  AcceleratorAssignment operator+=(AcceleratorExpr) const;

  friend bool operator==(const TensorObject& a, const TensorObject& b);

  friend bool operator<(const TensorObject& a, const TensorObject& b);

private:
  struct Content;
  std::shared_ptr<Content> content;

};

std::ostream& operator<<(std::ostream&, const TensorObject&);


class DynamicExpr : public util::IntrusivePtr<const DynamicExprNode> {
public:
  DynamicExpr() : util::IntrusivePtr<const DynamicExprNode>(nullptr) {}
  DynamicExpr(const DynamicExprNode* n) : util::IntrusivePtr<const DynamicExprNode>(n) {}

  DynamicExpr(int num);
  DynamicExpr(IndexVar i);

  void accept(DynamicExprVisitorStrict *) const;
  friend std::ostream& operator<<(std::ostream&, const DynamicExpr&);

};

DynamicExpr operator+(const DynamicExpr&, const DynamicExpr&);
DynamicExpr operator-(const DynamicExpr&, const DynamicExpr&);
DynamicExpr operator*(const DynamicExpr&, const DynamicExpr&);
DynamicExpr operator/(const DynamicExpr&, const DynamicExpr&);


template <typename SubType> bool isa(DynamicExpr);
template <typename SubType> SubType to(DynamicExpr);

class DynamicStmt : public util::IntrusivePtr<const DynamicStmtNode> {
public:
  DynamicStmt() : util::IntrusivePtr<const DynamicStmtNode>(nullptr) {}
  DynamicStmt(const DynamicStmtNode* n) : util::IntrusivePtr<const DynamicStmtNode>(n) {}

  void accept(DynamicStmtVisitorStrict *) const;
  friend std::ostream& operator<<(std::ostream&, const DynamicStmt&);

};

template <typename SubType> bool isa(DynamicStmt);
template <typename SubType> SubType to(DynamicStmt);

DynamicStmt operator==(const DynamicExpr&, const DynamicExpr&);
DynamicStmt operator!=(const DynamicExpr&, const DynamicExpr&);
DynamicStmt operator>(const DynamicExpr&, const DynamicExpr&);
DynamicStmt operator<(const DynamicExpr&, const DynamicExpr&);
DynamicStmt operator<=(const DynamicExpr&, const DynamicExpr&);
DynamicStmt operator>=(const DynamicExpr&, const DynamicExpr&);
DynamicStmt operator&&(const DynamicStmt&, const DynamicStmt&);
DynamicStmt operator||(const DynamicStmt&, const DynamicStmt&);

DynamicStmt forall(const DynamicIndexIterator&, const DynamicStmt&);
DynamicStmt exists(const DynamicIndexIterator&, const DynamicStmt&);

class DynamicIndexIterator : public DynamicExpr {
public:
  DynamicIndexIterator();
  DynamicIndexIterator(const DynamicIndexIteratorNode*);
  DynamicIndexIterator(DynamicOrder dynamicOrder);
  const DynamicOrder * getDynamicOrderPtr() const;

  friend bool operator==(const DynamicIndexIterator& a, const DynamicIndexIterator& b);
  friend bool operator<(const DynamicIndexIterator& a, const DynamicIndexIterator& b);

  DynamicOrder getDynamicOrder() const;
  typedef DynamicIndexIteratorNode Node;
};



class DynamicLiteral : public DynamicExpr {
public:
  DynamicLiteral();
  DynamicLiteral(const DynamicLiteralNode*);
  DynamicLiteral(int num);

  int getVal() const;
  typedef DynamicLiteralNode Node;
};


class DynamicIndexAccess : public DynamicExpr{
public:
  DynamicIndexAccess();
  DynamicIndexAccess(const DynamicIndexAccessNode*);
  DynamicIndexAccess(DynamicIndexIterator DynamicIndexIterator);

  DynamicIndexIterator getIterator() const;
  typedef DynamicIndexAccessNode Node;
};

class DynamicIndexMulInternal : public DynamicExpr {
  public: 
    DynamicIndexMulInternal();
    DynamicIndexMulInternal(const DynamicIndexMulInternalNode*);
    DynamicIndexMulInternal(DynamicOrder dynamicOrder);

    DynamicOrder getDynamicOrder() const;
    typedef DynamicIndexMulInternalNode Node;
};

class DynamicIndexLen : public DynamicExpr {
  public: 
    DynamicIndexLen();
    DynamicIndexLen(const DynamicIndexLenNode*);
    DynamicIndexLen(DynamicOrder dynamicOrder);

    DynamicOrder getDynamicOrder() const;
    typedef DynamicIndexLenNode Node;
};

class DynamicAdd : public DynamicExpr {
public:
  DynamicAdd();
  DynamicAdd(const DynamicAddNode*);
  DynamicAdd(DynamicExpr a, DynamicExpr b);

  DynamicExpr getA() const;
  DynamicExpr getB() const;

  typedef DynamicAddNode Node;
};

class DynamicMul : public DynamicExpr {
public:
  DynamicMul();
  DynamicMul(const DynamicMulNode*);
  DynamicMul(DynamicExpr a, DynamicExpr b);

  DynamicExpr getA() const;
  DynamicExpr getB() const;

  typedef DynamicMulNode Node;
};

class DynamicDiv : public DynamicExpr {
public:
  DynamicDiv();
  DynamicDiv(const DynamicDivNode*);
  DynamicDiv(DynamicExpr a, DynamicExpr b);

  DynamicExpr getA() const;
  DynamicExpr getB() const;

  typedef DynamicDivNode Node;
};

class DynamicMod : public DynamicExpr {
public:
  DynamicMod();
  DynamicMod(const DynamicModNode*);
  DynamicMod(DynamicExpr a, DynamicExpr b);

  DynamicExpr getA() const;
  DynamicExpr getB() const;

  typedef DynamicModNode Node;
};

class DynamicSub : public DynamicExpr {
public:
  DynamicSub();
  DynamicSub(const DynamicSubNode*);
  DynamicSub(DynamicExpr a, DynamicExpr b);

  DynamicExpr getA() const;
  DynamicExpr getB() const;

  typedef DynamicSubNode Node;
};

class DynamicIndexVar : public DynamicExpr {
public:
  DynamicIndexVar();
  DynamicIndexVar(const DynamicIndexVarNode*);
  DynamicIndexVar(IndexVar i);

  IndexVar getIVar() const;

  typedef DynamicIndexVarNode Node;
};


struct DynamicEqual : public DynamicStmt {
  public:
    DynamicEqual();
    DynamicEqual(const DynamicEqualNode*);
    DynamicEqual(DynamicExpr a, DynamicExpr b);

    DynamicExpr getA() const;
    DynamicExpr getB() const;

    typedef DynamicEqualNode Node;

};

struct DynamicNotEqual : public DynamicStmt {
  public:
    DynamicNotEqual();
    DynamicNotEqual(const DynamicNotEqualNode*);
    DynamicNotEqual(DynamicExpr a, DynamicExpr b);

    DynamicExpr getA() const;
    DynamicExpr getB() const;

    typedef DynamicNotEqualNode Node;

};

struct DynamicGreater : public DynamicStmt {
  public:
    DynamicGreater();
    DynamicGreater(const DynamicGreaterNode*);
    DynamicGreater(DynamicExpr a, DynamicExpr b);

    DynamicExpr getA() const;
    DynamicExpr getB() const;

    typedef DynamicGreaterNode Node;

};

struct DynamicLess: public DynamicStmt {
  public:
    DynamicLess();
    DynamicLess(const DynamicLessNode*);
    DynamicLess(DynamicExpr a, DynamicExpr b);

    DynamicExpr getA() const;
    DynamicExpr getB() const;

    typedef DynamicLessNode Node;

};

struct DynamicGeq: public DynamicStmt {
  public:
    DynamicGeq();
    DynamicGeq(const DynamicGeqNode*);
    DynamicGeq(DynamicExpr a, DynamicExpr b);

    DynamicExpr getA() const;
    DynamicExpr getB() const;

    typedef DynamicGeqNode Node;

};

struct DynamicLeq: public DynamicStmt {
  public:
    DynamicLeq();
    DynamicLeq(const DynamicLeqNode*);
    DynamicLeq(DynamicExpr a, DynamicExpr b);

    DynamicExpr getA() const;
    DynamicExpr getB() const;

    typedef DynamicLeqNode Node;

};

struct DynamicForall: public DynamicStmt {
  public:
    DynamicForall();
    DynamicForall(const DynamicForallNode*);
    DynamicForall(DynamicIndexIterator it, DynamicStmt stmt);

    DynamicIndexIterator getIterator() const;
    DynamicStmt getStmt() const;

    typedef DynamicForallNode Node;

};

struct DynamicExists: public DynamicStmt {
  public:
    DynamicExists();
    DynamicExists(const DynamicExistsNode*);
    DynamicExists(DynamicIndexIterator it, DynamicStmt stmt);

    DynamicIndexIterator getIterator() const;
    DynamicStmt getStmt() const;

    typedef DynamicExistsNode Node;

};

struct DynamicAnd: public DynamicStmt {
  public:
    DynamicAnd();
    DynamicAnd(const DynamicAndNode*);
    DynamicAnd(DynamicStmt a, DynamicStmt b);

    DynamicStmt getA() const;
    DynamicStmt getB() const;
    
    typedef DynamicAndNode Node;

};

struct DynamicOr: public DynamicStmt {
  public:
    DynamicOr();
    DynamicOr(const DynamicOrNode*);
    DynamicOr(DynamicStmt a, DynamicStmt b);

    DynamicStmt getA() const;
    DynamicStmt getB() const;
    
    typedef DynamicOrNode Node;

};

class PropertyExpr : public util::IntrusivePtr<const PropertyExprNode> {
public:
  PropertyExpr() : util::IntrusivePtr<const PropertyExprNode>(nullptr) {}
  PropertyExpr(const PropertyExprNode* n) : util::IntrusivePtr<const PropertyExprNode>(n) {}

  PropertyExpr(std::string property);
  // DynamicExpr(IndexVar i);

  void accept(PropertyExprVisitorStrict *) const;
  friend std::ostream& operator<<(std::ostream&, const PropertyExpr&);

};

PropertyExpr operator+(const PropertyExpr&, const PropertyExpr&);
PropertyExpr operator-(const PropertyExpr&, const PropertyExpr&);
PropertyExpr operator*(const PropertyExpr&, const PropertyExpr&);
PropertyExpr operator/(const PropertyExpr&, const PropertyExpr&);

template <typename SubType> bool isa(PropertyExpr);
template <typename SubType> SubType to(PropertyExpr);

struct PropertyTag: public PropertyExpr {
  public:
    PropertyTag();
    PropertyTag(const PropertyTagNode*);
    PropertyTag(const std::string& tag);

    std::string getTag() const;

    PropertyAssign operator=(const PropertyExpr&);
    PropertyAssign operator=(const PropertyExpr&) const;
    
    typedef PropertyTagNode Node;

};

struct PropertyAdd: public PropertyExpr {
  public:
    PropertyAdd();
    PropertyAdd(const PropertyAddNode*);
    PropertyAdd(PropertyExpr a, PropertyExpr b);

    PropertyExpr getA() const;
    PropertyExpr getB() const;
    
    typedef PropertyAddNode Node;
};

struct PropertySub: public PropertyExpr {
  public:
    PropertySub();
    PropertySub(const PropertySubNode*);
    PropertySub(PropertyExpr a, PropertyExpr b);

    PropertyExpr getA() const;
    PropertyExpr getB() const;
    
    typedef PropertySubNode Node;
};

struct PropertyMul: public PropertyExpr {
  public:
    PropertyMul();
    PropertyMul(const PropertyMulNode*);
    PropertyMul(PropertyExpr a, PropertyExpr b);

    PropertyExpr getA() const;
    PropertyExpr getB() const;
    
    typedef PropertyMulNode Node;
};

struct PropertyDiv: public PropertyExpr {
  public:
    PropertyDiv();
    PropertyDiv(const PropertyDivNode*);
    PropertyDiv(PropertyExpr a, PropertyExpr b);

    PropertyExpr getA() const;
    PropertyExpr getB() const;
    
    typedef PropertyDivNode Node;
};

class PropertyStmt : public util::IntrusivePtr<const PropertyStmtNode> {
public:
  PropertyStmt() : util::IntrusivePtr<const PropertyStmtNode>(nullptr) {}
  PropertyStmt(const PropertyStmtNode* n) : util::IntrusivePtr<const PropertyStmtNode>(n) {}

  // DynamicExpr(int num);
  // DynamicExpr(IndexVar i);

  void accept(PropertyStmtVisitorStrict *) const;
  friend std::ostream& operator<<(std::ostream&, const PropertyStmt&);

};

template <typename SubType> bool isa(PropertyStmt);
template <typename SubType> SubType to(PropertyStmt);

struct PropertyAssign: public PropertyStmt {
  public:
    PropertyAssign();
    PropertyAssign(const PropertyAssignNode*);
    PropertyAssign(PropertyTag rhs, PropertyExpr lhs);

    PropertyTag getLhs() const;
    PropertyExpr getRhs() const;
    
    typedef PropertyAssignNode Node;
};

}

#endif 