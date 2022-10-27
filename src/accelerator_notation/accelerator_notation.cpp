#include "taco/index_notation/index_notation.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <utility>
#include <functional>
#include <set>
#include <taco/ir/simplify.h>
#include "lower/mode_access.h"

#include "error/error_checks.h"
#include "taco/error/error_messages.h"
#include "taco/type.h"
#include "taco/format.h"

#include "taco/index_notation/properties.h"
#include "taco/index_notation/intrinsic.h"
#include "taco/index_notation/schedule.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/index_notation/transformations.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/accelerator_notation/accelerator_notation_nodes.h"
#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/accelerator_notation/accelerator_notation_printer.h"
#include "taco/accelerator_notation/accelerate_search.h"
#include "taco/accelerator_notation/accelerator_notation_rewriter.h"
#include "taco/ir/ir.h"
#include "taco/codegen/module.h"
#include "taco/tensor.h"

#include "taco/util/name_generator.h"
#include "taco/util/scopedset.h"
#include "taco/util/scopedmap.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"
#include "taco/util/functions.h"
#include "taco/util/env.h"



using namespace std;

namespace taco {

class TensorObject;

AcceleratorExpr::AcceleratorExpr(TensorObject tensorObject) 
    : AcceleratorExpr(new AcceleratorAccessNode(tensorObject,{},false)) { }

void AcceleratorExpr::accept(AcceleratorExprVisitorStrict *v) const {
  ptr->accept(v);
}

AcceleratorExpr::AcceleratorExpr(char val) : AcceleratorExpr(new AcceleratorLiteralNode(val)) {
}

AcceleratorExpr::AcceleratorExpr(int8_t val) : AcceleratorExpr(new AcceleratorLiteralNode(val)) {
}

AcceleratorExpr::AcceleratorExpr(int16_t val) : AcceleratorExpr(new AcceleratorLiteralNode(val)) {
}

AcceleratorExpr::AcceleratorExpr(int32_t val) : AcceleratorExpr(new AcceleratorLiteralNode(val)) {
}

AcceleratorExpr::AcceleratorExpr(int64_t val) : AcceleratorExpr(new AcceleratorLiteralNode(val)) {
}

AcceleratorExpr::AcceleratorExpr(uint8_t val) : AcceleratorExpr(new AcceleratorLiteralNode(val)) {
}

AcceleratorExpr::AcceleratorExpr(uint16_t val) : AcceleratorExpr(new AcceleratorLiteralNode(val)) {
}

AcceleratorExpr::AcceleratorExpr(uint32_t val) : AcceleratorExpr(new AcceleratorLiteralNode(val)) {
}

AcceleratorExpr::AcceleratorExpr(uint64_t val) : AcceleratorExpr(new AcceleratorLiteralNode(val)) {
}

AcceleratorExpr::AcceleratorExpr(float val) : AcceleratorExpr(new AcceleratorLiteralNode(val)) {
}

AcceleratorExpr::AcceleratorExpr(double val) : AcceleratorExpr(new AcceleratorLiteralNode(val)) {
}

AcceleratorExpr::AcceleratorExpr(std::complex<float> val) :AcceleratorExpr(new AcceleratorLiteralNode(val)){
}

AcceleratorExpr::AcceleratorExpr(std::complex<double> val) :AcceleratorExpr(new AcceleratorLiteralNode(val)){
}

Datatype AcceleratorExpr::getDataType() const {
  return const_cast<AcceleratorExprNode*>(this->ptr)->getDataType();
}

std::vector<IndexVar> AcceleratorExpr::getIndexVars() const{
  std::vector<IndexVar> vars;
  std::set<IndexVar> seen;

  acceleratorMatch(*this,
   std::function<void(const AcceleratorAccessNode*)>([&](const AcceleratorAccessNode* op) {
      for (auto& var : op->indexVars) {
        if (!util::contains(seen, var)) {
          vars.push_back(var);
          seen.insert(var);
        }
      }
    })
  );

  return vars;
}

std::map<IndexVar,Dimension> AcceleratorExpr::getIndexVarDomains() const {
  map<IndexVar, Dimension> indexVarDomains;
  acceleratorMatch(*this,
    std::function<void(const AcceleratorAccessNode*)>([&indexVarDomains](const AcceleratorAccessNode* op) {
      auto& type = op->tensorObject.getType();
      auto& vars = op->indexVars;
      for (size_t i = 0; i < vars.size(); i++) {
        if (!util::contains(indexVarDomains, vars[i])) {
          indexVarDomains.insert({vars[i], type.getShape().getDimension(i)});
        }
        else {
          taco_iassert(indexVarDomains.at(vars[i]) ==
                       type.getShape().getDimension(i))
              << "Index variable used to index incompatible dimensions";
        }
      }
    })
  );

  return indexVarDomains;

}

std::ostream& operator<<(std::ostream& os, const AcceleratorExpr& expr) {
  if (!expr.defined()) return os << "AcceleratorExpr()";
  AcceleratorNotationPrinter printer(os);
  printer.print(expr);
  return os;
}

AcceleratorExpr operator-(const AcceleratorExpr& expr) {
  return new AcceleratorNegNode(expr.ptr);
}

AcceleratorExpr operator+(const AcceleratorExpr& lhs, const AcceleratorExpr& rhs) {
  return new AcceleratorAddNode(lhs, rhs);
}

AcceleratorExpr operator-(const AcceleratorExpr& lhs, const AcceleratorExpr& rhs) {
  return new AcceleratorSubNode(lhs, rhs);
}

AcceleratorExpr operator*(const AcceleratorExpr& lhs, const AcceleratorExpr& rhs) {
  return new AcceleratorMulNode(lhs, rhs);
}

AcceleratorExpr operator/(const AcceleratorExpr& lhs, const AcceleratorExpr& rhs) {
  return new AcceleratorDivNode(lhs, rhs);
}


void AcceleratorStmt::accept(AcceleratorStmtVisitorStrict *v) const {
  ptr->accept(v);
}

std::vector<IndexVar> AcceleratorStmt::getIndexVars() const{
  std::vector<IndexVar> vars;
  std::set<IndexVar> seen;

  acceleratorMatch(*this,
    std::function<void(const AcceleratorAssignmentNode*,AcceleratorMatcher*)>([&](
        const AcceleratorAssignmentNode* op, AcceleratorMatcher* ctx) {
      for (auto& var : op->lhs.getIndexVars()) {
        if (!util::contains(seen, var)) {
          vars.push_back(var);
          seen.insert(var);
        }
      }
      ctx->acceleratorMatch(op->rhs);
   }),
   std::function<void(const AcceleratorAccessNode*)>([&](const AcceleratorAccessNode* op) {
      for (auto& var : op->indexVars) {
        if (!util::contains(seen, var)) {
          vars.push_back(var);
          seen.insert(var);
        }
      }
    })
  );

  return vars;

}

std::ostream& operator<<(std::ostream& os, const AcceleratorStmt& expr) {
  if (!expr.defined()) return os << "IndexStmt()";
  AcceleratorNotationPrinter printer(os);
  printer.print(expr);
  return os;
}

AcceleratorAccess::AcceleratorAccess(const AcceleratorAccessNode* n) : AcceleratorExpr(n) {
}

AcceleratorAccess::AcceleratorAccess(const TensorObject& tensor, const std::vector<IndexVar>& indices,
               bool isAccessingStructure)
    : AcceleratorAccess(new AcceleratorAccessNode(tensor, indices, isAccessingStructure)) {
}

const TensorObject& AcceleratorAccess::getTensorObject() const {
  return getNode(*this)->tensorObject;
}

const std::vector<IndexVar>& AcceleratorAccess::getIndexVars() const {
  return getNode(*this)->indexVars;
}

AcceleratorAssignment AcceleratorAccess::operator=(const AcceleratorExpr& expr){
  AcceleratorAssignment assignment = AcceleratorAssignment(*this, expr);
  //TODO: do some type checking
  return assignment;
}

AcceleratorAssignment AcceleratorAccess::operator=(const AcceleratorExpr& expr) const{
  AcceleratorAssignment assignment = AcceleratorAssignment(*this, expr);
  //TODO: do some type checking
  return assignment;
}


/// Must override the default Access operator=, otherwise it is a copy.
AcceleratorAssignment AcceleratorAccess::operator=(const AcceleratorAccess& access){
  return operator=(static_cast<AcceleratorExpr>(access));
}

AcceleratorAssignment AcceleratorAccess::operator=(const AcceleratorAccess& access) const{
  return operator=(static_cast<AcceleratorExpr>(access));
}

/// Must disambiguate TensorVar as it can be implicitly converted to IndexExpr
/// and AccesExpr.
AcceleratorAssignment AcceleratorAccess::operator=(const TensorObject& tensor){
  return operator=(AcceleratorAccess(tensor));
}

AcceleratorAssignment AcceleratorAccess::operator+=(const AcceleratorExpr& expr){
  TensorObject result = getTensorObject();
  AcceleratorAssignment assignment = AcceleratorAssignment(
    result,
    getIndexVars(),
    expr,
    AcceleratorAdd()
  );
  return assignment;
}

template <> bool isa<AcceleratorAccess>(AcceleratorExpr e) {
  return isa<AcceleratorAccessNode>(e.ptr);
}

template <> AcceleratorAccess to<AcceleratorAccess>(AcceleratorExpr e) {
  taco_iassert(isa<AcceleratorAccess>(e));
  return AcceleratorAccess(to<AcceleratorAccessNode>(e.ptr));
}


AcceleratorDynamicIndex::AcceleratorDynamicIndex(const AcceleratorDynamicIndexNode* n) : AcceleratorExpr(n){
}
AcceleratorDynamicIndex::AcceleratorDynamicIndex(const TensorObject& tensorObject, const std::vector<IndexObject>& indices)
: AcceleratorDynamicIndex(new AcceleratorDynamicIndexNode(tensorObject, indices)){
}

AcceleratorAssignment AcceleratorDynamicIndex::operator=(const AcceleratorExpr& expr){
  AcceleratorAssignment assignment = AcceleratorAssignment(*this, expr);
  return assignment;
}

AcceleratorAssignment AcceleratorDynamicIndex::operator=(const AcceleratorExpr& expr) const{
  AcceleratorAssignment assignment = AcceleratorAssignment(*this, expr);
  return assignment;
}

AcceleratorAssignment AcceleratorDynamicIndex::operator=(const AcceleratorAccess& expr){
  AcceleratorAssignment assignment = AcceleratorAssignment(*this, expr);
  return assignment;
}

AcceleratorAssignment AcceleratorDynamicIndex::operator=(const AcceleratorAccess& expr) const{
  return operator=(static_cast<AcceleratorExpr>(expr));
}

AcceleratorAssignment AcceleratorDynamicIndex::operator=(const AcceleratorDynamicIndex& expr){
  AcceleratorAssignment assignment = AcceleratorAssignment(*this, expr);
  return assignment;
}


AcceleratorAssignment AcceleratorDynamicIndex::operator=(const AcceleratorDynamicIndex& expr) const{
  AcceleratorAssignment assignment = AcceleratorAssignment(*this, expr);
  return assignment;
}

AcceleratorAssignment AcceleratorDynamicIndex::operator=(const TensorObject& t){
  return operator=(static_cast<AcceleratorExpr>(AcceleratorAccess(t)));
}

AcceleratorAssignment AcceleratorDynamicIndex::operator+=(const AcceleratorExpr& expr){
  AcceleratorAssignment assignment = AcceleratorAssignment(*this, expr, AcceleratorAdd());
  return assignment;
}


const TensorObject& AcceleratorDynamicIndex::getTensorObject() const{
  return getNode(*this)->t;
}
const std::vector<IndexObject>& AcceleratorDynamicIndex::getIndexObjects() const{
   return getNode(*this)->indexObject;
}

template <> bool isa<AcceleratorDynamicIndex>(AcceleratorExpr e) {
  return isa<AcceleratorDynamicIndexNode>(e.ptr);
}

template <> AcceleratorDynamicIndex to<AcceleratorDynamicIndex>(AcceleratorExpr e) {
  taco_iassert(isa<AcceleratorDynamicIndex>(e));
  return AcceleratorDynamicIndex(to<AcceleratorDynamicIndexNode>(e.ptr));
}


AcceleratorLiteral::AcceleratorLiteral(const AcceleratorLiteralNode* n) : AcceleratorExpr(n) {
}

AcceleratorLiteral::AcceleratorLiteral(bool val) : AcceleratorLiteral(new AcceleratorLiteralNode(val)) {
}

AcceleratorLiteral::AcceleratorLiteral(unsigned char val) : AcceleratorLiteral(new AcceleratorLiteralNode(val)) {
}

AcceleratorLiteral::AcceleratorLiteral(unsigned short val) : AcceleratorLiteral(new AcceleratorLiteralNode(val)) {
}

AcceleratorLiteral::AcceleratorLiteral(unsigned int val) : AcceleratorLiteral(new AcceleratorLiteralNode(val)) {
}

AcceleratorLiteral::AcceleratorLiteral(unsigned long val) : AcceleratorLiteral(new AcceleratorLiteralNode(val)) {
}

AcceleratorLiteral::AcceleratorLiteral(unsigned long long val) : AcceleratorLiteral(new AcceleratorLiteralNode(val)) {
}

AcceleratorLiteral::AcceleratorLiteral(char val) : AcceleratorLiteral(new AcceleratorLiteralNode(val)) {
}

AcceleratorLiteral::AcceleratorLiteral(short val) : AcceleratorLiteral(new AcceleratorLiteralNode(val)) {
}

AcceleratorLiteral::AcceleratorLiteral(int val) : AcceleratorLiteral(new AcceleratorLiteralNode(val)) {
}

AcceleratorLiteral::AcceleratorLiteral(long val) : AcceleratorLiteral(new AcceleratorLiteralNode(val)) {
}

AcceleratorLiteral::AcceleratorLiteral(long long val) : AcceleratorLiteral(new AcceleratorLiteralNode(val)) {
}

AcceleratorLiteral::AcceleratorLiteral(int8_t val) : AcceleratorLiteral(new AcceleratorLiteralNode(val)) {
}

AcceleratorLiteral::AcceleratorLiteral(float val) : AcceleratorLiteral(new AcceleratorLiteralNode(val)) {
}

AcceleratorLiteral::AcceleratorLiteral(double val) : AcceleratorLiteral(new AcceleratorLiteralNode(val)) {
}

AcceleratorLiteral::AcceleratorLiteral(std::complex<float> val) : AcceleratorLiteral(new AcceleratorLiteral(val)) {
}

AcceleratorLiteral::AcceleratorLiteral(std::complex<double> val) : AcceleratorLiteral(new AcceleratorLiteral(val)) {
}

AcceleratorLiteral AcceleratorLiteral::zero(Datatype type) {
  switch (type.getKind()) {
    case Datatype::Bool:        return AcceleratorLiteral(false);
    case Datatype::UInt8:       return AcceleratorLiteral(uint8_t(0));
    case Datatype::UInt16:      return AcceleratorLiteral(uint16_t(0));
    case Datatype::UInt32:      return AcceleratorLiteral(uint32_t(0));
    case Datatype::UInt64:      return AcceleratorLiteral(uint64_t(0));
    case Datatype::Int8:        return AcceleratorLiteral(int8_t(0));
    case Datatype::Int16:       return AcceleratorLiteral(int16_t(0));
    case Datatype::Int32:       return AcceleratorLiteral(int32_t(0));
    case Datatype::Int64:       return AcceleratorLiteral(int64_t(0));
    case Datatype::Float32:     return AcceleratorLiteral(float(0.0));
    case Datatype::Float64:     return AcceleratorLiteral(double(0.0));
    case Datatype::Complex64:   return AcceleratorLiteral(std::complex<float>());
    case Datatype::Complex128:  return AcceleratorLiteral(std::complex<double>());
    default:                    taco_ierror << "unsupported type";
  };

  return AcceleratorLiteral();
}

template <typename T> T AcceleratorLiteral::getVal() const {
  return getNode(*this)->getVal<T>();
}

void* AcceleratorLiteral::getValPtr() {
  return getNode(*this)->val;
}

template <> bool isa<AcceleratorLiteral>(AcceleratorExpr e) {
  return isa<AcceleratorLiteralNode>(e.ptr);
}

template <> AcceleratorLiteral to<AcceleratorLiteral>(AcceleratorExpr e) {
  taco_iassert(isa<AcceleratorLiteral>(e));
  return AcceleratorLiteral(to<AcceleratorLiteralNode>(e.ptr));
}

//class AcceleratorNegNode
AcceleratorNeg::AcceleratorNeg(const AcceleratorNegNode* n) : AcceleratorExpr(n) {
}

AcceleratorNeg::AcceleratorNeg(AcceleratorExpr a) : AcceleratorNeg(new AcceleratorNegNode(a)) {
}

AcceleratorExpr AcceleratorNeg::getA() const {
  return getNode(*this)->a;
}

template <> bool isa<AcceleratorNeg>(AcceleratorExpr e) {
  return isa<AcceleratorNegNode>(e.ptr);
}

template <> AcceleratorNeg to<AcceleratorNeg>(AcceleratorExpr e) {
  taco_iassert(isa<AcceleratorNeg>(e));
  return AcceleratorNeg(to<AcceleratorNegNode>(e.ptr));
}


// class AcceleratorAdd
AcceleratorAdd::AcceleratorAdd() : AcceleratorAdd(new AcceleratorAddNode) {
}

AcceleratorAdd::AcceleratorAdd(const AcceleratorAddNode* n) : AcceleratorExpr(n) {
}

AcceleratorAdd::AcceleratorAdd(AcceleratorExpr a, AcceleratorExpr b) : AcceleratorAdd(new AcceleratorAddNode(a, b)) {
}

AcceleratorExpr AcceleratorAdd::getA() const {
  return getNode(*this)->a;
}

AcceleratorExpr AcceleratorAdd::getB() const {
  return getNode(*this)->b;
}

template <> bool isa<AcceleratorAdd>(AcceleratorExpr e) {
  return isa<AcceleratorAddNode>(e.ptr);
}

template <> AcceleratorAdd to<AcceleratorAdd>(AcceleratorExpr e) {
  taco_iassert(isa<AcceleratorAdd>(e));
  return AcceleratorAdd(to<AcceleratorAddNode>(e.ptr));
}

// class AcceleratorSub
AcceleratorSub::AcceleratorSub() : AcceleratorSub(new AcceleratorSubNode) {
}

AcceleratorSub::AcceleratorSub(const AcceleratorSubNode* n) : AcceleratorExpr(n) {
}

AcceleratorSub::AcceleratorSub(AcceleratorExpr a, AcceleratorExpr b) : AcceleratorSub(new AcceleratorSubNode(a, b)) {
}

AcceleratorExpr AcceleratorSub::getA() const {
  return getNode(*this)->a;
}

AcceleratorExpr AcceleratorSub::getB() const {
  return getNode(*this)->b;
}

template <> bool isa<AcceleratorSub>(AcceleratorExpr e) {
  return isa<AcceleratorSubNode>(e.ptr);
}

template <> AcceleratorSub to<AcceleratorSub>(AcceleratorExpr e) {
  taco_iassert(isa<AcceleratorSub>(e));
  return AcceleratorSub(to<AcceleratorSubNode>(e.ptr));
}

// class AcceleratorMul
AcceleratorMul::AcceleratorMul() : AcceleratorMul(new AcceleratorMulNode) {
}

AcceleratorMul::AcceleratorMul(const AcceleratorMulNode* n) : AcceleratorExpr(n) {
}

AcceleratorMul::AcceleratorMul(AcceleratorExpr a, AcceleratorExpr b) : AcceleratorMul(new AcceleratorMulNode(a, b)) {
}

AcceleratorExpr AcceleratorMul::getA() const {
  return getNode(*this)->a;
}

AcceleratorExpr AcceleratorMul::getB() const {
  return getNode(*this)->b;
}

template <> bool isa<AcceleratorMul>(AcceleratorExpr e) {
  return isa<AcceleratorMulNode>(e.ptr);
}

template <> AcceleratorMul to<AcceleratorMul>(AcceleratorExpr e) {
  taco_iassert(isa<AcceleratorMul>(e));
  return AcceleratorMul(to<AcceleratorMulNode>(e.ptr));
}

// class AcceleratorDiv
AcceleratorDiv::AcceleratorDiv() : AcceleratorDiv(new AcceleratorDivNode) {
}

AcceleratorDiv::AcceleratorDiv(const AcceleratorDivNode* n) : AcceleratorExpr(n) {
}

AcceleratorDiv::AcceleratorDiv(AcceleratorExpr a, AcceleratorExpr b) : AcceleratorDiv(new AcceleratorDivNode(a, b)) {
}

AcceleratorExpr AcceleratorDiv::getA() const {
  return getNode(*this)->a;
}

AcceleratorExpr AcceleratorDiv::getB() const {
  return getNode(*this)->b;
}

template <> bool isa<AcceleratorDiv>(AcceleratorExpr e) {
  return isa<AcceleratorDivNode>(e.ptr);
}

template <> AcceleratorDiv to<AcceleratorDiv>(AcceleratorExpr e) {
  taco_iassert(isa<AcceleratorDiv>(e));
  return AcceleratorDiv(to<AcceleratorDivNode>(e.ptr));
}

//class AcceleratorSqrt
AcceleratorSqrt::AcceleratorSqrt(const AcceleratorSqrtNode* n) : AcceleratorExpr(n) {
}

AcceleratorSqrt::AcceleratorSqrt(AcceleratorExpr a) : AcceleratorSqrt(new AcceleratorSqrtNode(a)) {
}

AcceleratorExpr AcceleratorSqrt::getA() const {
  return getNode(*this)->a;
}

template <> bool isa<AcceleratorSqrt>(AcceleratorExpr e) {
  return isa<AcceleratorSqrtNode>(e.ptr);
}

template <> AcceleratorSqrt to<AcceleratorSqrt>(AcceleratorExpr e) {
  taco_iassert(isa<AcceleratorSqrt>(e));
  return AcceleratorSqrt(to<AcceleratorSqrtNode>(e.ptr));
}

//class AcceleratorReduction
AcceleratorReduction::AcceleratorReduction(const AcceleratorReductionNode* n) : AcceleratorExpr(n){
}

AcceleratorReduction::AcceleratorReduction(AcceleratorExpr op, IndexVar var, AcceleratorExpr expr) : AcceleratorReduction(new AcceleratorReductionNode(op, var, expr)) {
}

AcceleratorExpr AcceleratorReduction::getOp() const{
  return getNode(*this)->op;
}

IndexVar AcceleratorReduction::getVar() const{
  return getNode(*this)->var;
}

AcceleratorExpr AcceleratorReduction::getExpr() const{
  return getNode(*this)->a;
}

AcceleratorReduction sum(IndexVar i, AcceleratorExpr expr){
  return AcceleratorReduction(new AcceleratorAddNode, i, expr);
}

template <> bool isa<AcceleratorReduction>(AcceleratorExpr s) {
  return isa<AcceleratorReductionNode>(s.ptr);
}

template <> AcceleratorReduction to<AcceleratorReduction>(AcceleratorExpr s) {
  taco_iassert(isa<AcceleratorReduction>(s));
  return AcceleratorReduction(to<AcceleratorReductionNode>(s.ptr));
}

//class AcceleratorForall
AcceleratorForall::AcceleratorForall(const AcceleratorForallNode* n) : AcceleratorStmt(n) {}

AcceleratorForall::AcceleratorForall(IndexVar indexVar, AcceleratorStmt stmt) : AcceleratorForall(new AcceleratorForallNode(indexVar, stmt)) {}

IndexVar AcceleratorForall::getIndexVar() const{
  return getNode(*this)->indexVar;
}

AcceleratorStmt AcceleratorForall::getStmt() const{
  return getNode(*this)->stmt;
}

AcceleratorForall forall(IndexVar i, AcceleratorStmt stmt){
  return AcceleratorForall(i, stmt);
}

template <> bool isa<AcceleratorForall>(AcceleratorStmt s) {
  return isa<AcceleratorForallNode>(s.ptr);
}

template <> AcceleratorForall to<AcceleratorForall>(AcceleratorStmt s) {
  taco_iassert(isa<AcceleratorForall>(s));
  return AcceleratorForall(to<AcceleratorForallNode>(s.ptr));
}

//class AcceleratorAssigment
AcceleratorAssignment::AcceleratorAssignment(const AcceleratorAssignmentNode* n) : AcceleratorStmt(n) {
}

AcceleratorAssignment::AcceleratorAssignment(AcceleratorAccess lhs, AcceleratorExpr rhs, AcceleratorExpr op)
    : AcceleratorAssignment(new AcceleratorAssignmentNode(lhs, rhs, op)) {
}

AcceleratorAssignment::AcceleratorAssignment(AcceleratorDynamicIndex lhs, AcceleratorExpr rhs, AcceleratorExpr op)
    : AcceleratorAssignment(new AcceleratorAssignmentNode(lhs, rhs, op)) {
}

AcceleratorAssignment::AcceleratorAssignment(TensorObject tensor, std::vector<IndexVar> indices, AcceleratorExpr rhs,
             AcceleratorExpr op)
      :  AcceleratorAssignment(AcceleratorAccess(tensor, indices), rhs, op) { 
}

/// Return the assignment's left-hand side.
AcceleratorAccess AcceleratorAssignment::getLhs() const {
  return getNode(*this)->lhs;
}

/// Return the assignment's right-hand side.
AcceleratorExpr AcceleratorAssignment::getRhs() const{
  return getNode(*this)->rhs;
}

/// Return the assignment compound operator (e.g., `+=`) or an undefined
/// expression if the assignment is not compound (`=`).
AcceleratorExpr AcceleratorAssignment::getOperator() const{
  return getNode(*this)->op;
}

const std::vector<IndexVar>& AcceleratorAssignment::getFreeVars() const {
  return getLhs().getIndexVars();
}

template <> bool isa<AcceleratorAssignment>(AcceleratorStmt s) {
  return isa<AcceleratorAssignmentNode>(s.ptr);
}

template <> AcceleratorAssignment to<AcceleratorAssignment>(AcceleratorStmt s) {
  taco_iassert(isa<AcceleratorAssignment>(s));
  return AcceleratorAssignment(to<AcceleratorAssignmentNode>(s.ptr));
}

const std::vector<IndexVar> AcceleratorAssignment::getImplicitReducionVars() const{
  std::vector<IndexVar> lhsVars = getLhs().getIndexVars();
  std::vector<IndexVar> rhsVars;
  std::set<IndexVar> seen; 

  acceleratorMatch(getRhs(),
   std::function<void(const AcceleratorAccessNode*)>([&](const AcceleratorAccessNode* op) {
      for (auto& var : op->indexVars) {
        if (!util::contains(seen, var)) {
          rhsVars.push_back(var);
          seen.insert(var);
        }
      }
    })
  );

  std::vector<IndexVar> reducedVars;
  for (auto rhsVar: rhsVars){
    if(!util::contains(lhsVars, rhsVar)) {
      reducedVars.push_back(rhsVar);
    }
  }
  
  return reducedVars;
}

static std::vector<IndexVar> getIndexVars(AcceleratorExpr expr){
  std::vector<IndexVar> vars;
  std::set<IndexVar> seen; 

  acceleratorMatch(expr,
   std::function<void(const AcceleratorAccessNode*)>([&](const AcceleratorAccessNode* op) {
      for (auto& var : op->indexVars) {
        if (!util::contains(seen, var)) {
          vars.push_back(var);
          seen.insert(var);
        }
      }
    })
  );

  return vars;

}


AcceleratorAssignment makeReductionNotation(AcceleratorAssignment assignment){
  AcceleratorExpr expr = assignment.getRhs();
  std::vector<IndexVar> free = assignment.getLhs().getIndexVars();

  struct MakeReductionNotation : AcceleratorNotationRewriter { 
    MakeReductionNotation(const vector<IndexVar>& free)
        : free(free.begin(), free.end()){}

    std::set<IndexVar> free;
    bool onlyOneTerm;

    AcceleratorExpr addReductions(AcceleratorExpr expr) {
      auto vars = getIndexVars(expr);
      for (auto& var : util::reverse(vars)) {
        if (!util::contains(free, var)) {
          expr = sum(var,expr);
        }
      }
      return expr;
    }

    AcceleratorExpr einsum(const AcceleratorExpr& expr) {
      onlyOneTerm = true;
      AcceleratorExpr einsumexpr = rewrite(expr);

      if (onlyOneTerm) {
        einsumexpr = addReductions(einsumexpr);
      }

      return einsumexpr;
    }

    using AcceleratorNotationRewriter::visit;

    void visit(const AcceleratorAddNode* op) {
      // Sum every reduction variables over each term
      onlyOneTerm = false;

      AcceleratorExpr a = addReductions(op->a);
      AcceleratorExpr b = addReductions(op->b);
      if (a == op->a && b == op->b) {
        expr = op;
      }
      else {
        expr = new AcceleratorAddNode(a, b);
      }
    }

    void visit(const AcceleratorSubNode* op) {
      // Sum every reduction variables over each term
      onlyOneTerm = false;

      AcceleratorExpr a = addReductions(op->a);
      AcceleratorExpr b = addReductions(op->b);
      if (a == op->a && b == op->b) {
        expr = op;
      }
      else {
        expr = new AcceleratorSubNode(a, b);
      }
    }
  };

  return AcceleratorAssignment(assignment.getLhs(),
                    MakeReductionNotation(free).einsum(expr),
                    assignment.getOperator());
}

// class TensorObject
struct TensorObject::Content {
  string name;
  Type type;
  Format format;
  Schedule schedule;
  Literal fill;
  std::set<std::string> properties;
};

static Format createDenseFormat(const Type& type) {
  return Format(vector<ModeFormatPack>(type.getOrder(), ModeFormat(Dense)));
}

TensorObject::TensorObject(const Type& type)
: TensorObject(util::uniqueName('A'), type, createDenseFormat(type)) {
}

TensorObject::TensorObject(const std::string& name, const Type& type)
: TensorObject(name, type, createDenseFormat(type)) {
}

TensorObject::TensorObject(const Type& type, const Format& format)
: TensorObject(util::uniqueName('A'), type, format) {
}

TensorObject::TensorObject(const std::string& name, const Type& type, const Format& format)
    : content(new Content) {
  
  content->name = name;
  content->type = type; 
  content->format = format;
}

TensorObject::TensorObject() : content(new Content) {
  content->name = util::uniqueName('A');
}

std::string TensorObject::getName() const {
  return content->name;
}

/// Sets the name of the tensor object.
void TensorObject::setName(const std::string& name) const{
  content->name = name;
}

/// Returns the type of the tensor variable.
const Type& TensorObject::getType() const{
  return content->type;
}

const Format& TensorObject::getFormat() const{
  return content->format;
}

int TensorObject::getOrder() const {
  return content->type.getShape().getOrder();
}

std::set<std::string> TensorObject::getProperties() const{
  return content->properties;
}

void TensorObject::addProperty(std::string property) const{
  content->properties.insert(property);
}

bool TensorObject::hasProperty(std::string property) const{
 return content->properties.count(property);
}

const AcceleratorAccess TensorObject::operator()(const std::vector<IndexVar>& indices) const {
  taco_uassert((int)indices.size() == getOrder()) <<
      "A tensor of order " << getOrder() << " must be indexed with " <<
      getOrder() << " variables, but is indexed with:  " << util::join(indices);
  return AcceleratorAccess(new AcceleratorAccessNode(*this, indices, false));
}

AcceleratorAccess TensorObject::operator()(const std::vector<IndexVar>& indices) {
  taco_uassert((int)indices.size() == getOrder()) <<
      "A tensor of order " << getOrder() << " must be indexed with " <<
      getOrder() << " variables, but is indexed with:  " << util::join(indices);
  return AcceleratorAccess(new AcceleratorAccessNode(*this, indices, false));
}

AcceleratorDynamicIndex TensorObject::operator[](const std::vector<IndexObject>& indices) const{
   return AcceleratorDynamicIndex(new AcceleratorDynamicIndexNode(*this, indices));
}

AcceleratorAssignment TensorObject::operator=(AcceleratorExpr expr) {
  taco_uassert(getOrder() == 0)
      << "Must use index variable on the left-hand-side when assigning an "
      << "expression to a non-scalar tensor.";
  AcceleratorAssignment assignment = AcceleratorAssignment(*this, {}, expr);
  //TODOD: Maybe add some check here
  return assignment;
}

AcceleratorAssignment TensorObject::operator=(AcceleratorExpr expr) const {
  taco_uassert(getOrder() == 0)
      << "Must use index variable on the left-hand-side when assigning an "
      << "expression to a non-scalar tensor.";
  AcceleratorAssignment assignment = AcceleratorAssignment(*this, {}, expr);
  //TODOD: Maybe add some check here
  return assignment;
}

AcceleratorAssignment TensorObject::operator+=(AcceleratorExpr expr) {
  taco_uassert(getOrder() == 0)
      << "Must use index variable on the left-hand-side when assigning an "
      << "expression to a non-scalar tensor.";
  AcceleratorAssignment assignment = AcceleratorAssignment(*this, {}, expr, new AcceleratorAddNode);
  return assignment;
}

AcceleratorAssignment TensorObject::operator+=(AcceleratorExpr expr) const {
  taco_uassert(getOrder() == 0)
      << "Must use index variable on the left-hand-side when assigning an "
      << "expression to a non-scalar tensor.";
  AcceleratorAssignment assignment = AcceleratorAssignment(*this, {}, expr, new AcceleratorAddNode);
  return assignment;
}

bool operator==(const TensorObject& a, const TensorObject& b) {
  return a.content == b.content;
}

bool operator<(const TensorObject& a, const TensorObject& b) {
  return a.content < b.content;
}

std::ostream& operator<<(std::ostream& os, const TensorObject& var){
  return os << var.getName() << " : " << var.getType();
}


std::ostream& operator<<(std::ostream& os, const DynamicExpr& expr) {
  if (!expr.defined()) return os << "DynamicExpr()";
  DynamicNotationPrinter printer(os);
  printer.print(expr);
  return os;
}

DynamicExpr::DynamicExpr(int val) : DynamicExpr(new DynamicLiteralNode(val)) {
}

DynamicExpr::DynamicExpr(IndexVar i) : DynamicExpr(new DynamicIndexVarNode(i)) {
}

void  DynamicExpr::accept(DynamicExprVisitorStrict *v) const {
  ptr->accept(v);
}

DynamicExpr operator+(const DynamicExpr& lhs, const DynamicExpr& rhs) {
  return new DynamicAddNode(lhs, rhs);
}

DynamicExpr operator-(const DynamicExpr& lhs, const DynamicExpr& rhs) {
  return new DynamicSubNode(lhs, rhs);
}

DynamicExpr operator*(const DynamicExpr& lhs, const DynamicExpr& rhs) {
  return new DynamicMulNode(lhs, rhs);
}

DynamicExpr operator/(const DynamicExpr& lhs, const DynamicExpr& rhs) {
  return new DynamicDivNode(lhs, rhs);
}

void  DynamicStmt::accept(DynamicStmtVisitorStrict *v) const {
  ptr->accept(v);
}

std::ostream& operator<<(std::ostream& os, const DynamicStmt& expr) {
  if (!expr.defined()) return os << "DynamicStmt()";
  DynamicNotationPrinter printer(os);
  printer.print(expr);
  return os;
}

DynamicStmt operator==(const DynamicExpr& a, const DynamicExpr& b){
  return DynamicEqual(a, b);
}

DynamicStmt operator!=(const DynamicExpr& a, const DynamicExpr& b){
  return DynamicNotEqual(a, b);
}

DynamicStmt operator>(const DynamicExpr& a, const DynamicExpr& b){
  return DynamicGreater(a, b);
}

DynamicStmt operator<(const DynamicExpr& a, const DynamicExpr& b){
  return DynamicLess(a, b);
}

DynamicStmt operator>=(const DynamicExpr& a, const DynamicExpr& b){
  return DynamicGeq(a, b);
}

DynamicStmt operator<=(const DynamicExpr& a, const DynamicExpr& b){
  return DynamicLeq(a, b);
}

DynamicStmt operator&&(const DynamicStmt& a, const DynamicStmt& b){
  return DynamicAnd(a, b);
}

DynamicStmt operator||(const DynamicStmt& a, const DynamicStmt& b){
  return DynamicOr(a, b);
}

DynamicStmt forall(const DynamicIndexIterator& it, const DynamicStmt& stmt){
  return DynamicForall(it, stmt);
}

DynamicStmt exists(const DynamicIndexIterator& it, const DynamicStmt& stmt){
  return DynamicExists(it, stmt);
}

DynamicIndexIterator::DynamicIndexIterator() :  DynamicIndexIterator(new DynamicIndexIteratorNode) {}
DynamicIndexIterator::DynamicIndexIterator(const DynamicIndexIteratorNode* n) : DynamicExpr(n){}
DynamicIndexIterator::DynamicIndexIterator(DynamicOrder dynamicOrder) : DynamicExpr(new DynamicIndexIteratorNode(dynamicOrder)) {}

DynamicOrder DynamicIndexIterator::getDynamicOrder() const{
   return  getNode(*this)->dynamicOrder;
}

const DynamicOrder * DynamicIndexIterator::getDynamicOrderPtr() const{
   return  &(getNode(*this)->dynamicOrder);
}

bool operator==(const DynamicIndexIterator& a, const DynamicIndexIterator& b){
  return a.ptr == b.ptr;
}

bool operator<(const DynamicIndexIterator& a, const DynamicIndexIterator& b){
  return a.ptr < b.ptr;
}

template <> bool isa<DynamicIndexIterator>(DynamicExpr e) {
  return isa<DynamicIndexIteratorNode>(e.ptr);
}

template <> DynamicIndexIterator to<DynamicIndexIterator>(DynamicExpr e) {
  taco_iassert(isa<DynamicIndexIterator>(e));
  return DynamicIndexIterator(to<DynamicIndexIteratorNode>(e.ptr));
}

DynamicLiteral::DynamicLiteral() : DynamicLiteral(new DynamicLiteralNode) {}
DynamicLiteral::DynamicLiteral(const DynamicLiteralNode* n) : DynamicExpr(n){}
DynamicLiteral::DynamicLiteral(int num) : DynamicExpr(new DynamicLiteralNode(num)) {}

int DynamicLiteral::getVal() const{
  return  getNode(*this)->num;
}

template <> bool isa<DynamicLiteral>(DynamicExpr e) {
  return isa<DynamicLiteralNode>(e.ptr);
}

template <> DynamicLiteral to<DynamicLiteral>(DynamicExpr e) {
  taco_iassert(isa<DynamicLiteral>(e));
  return DynamicLiteral(to<DynamicLiteralNode>(e.ptr));
}

DynamicIndexAccess::DynamicIndexAccess() :  DynamicIndexAccess(new DynamicIndexAccessNode) {}
DynamicIndexAccess::DynamicIndexAccess(const DynamicIndexAccessNode* n) : DynamicExpr(n){}
DynamicIndexAccess::DynamicIndexAccess(DynamicIndexIterator it) : DynamicExpr(new DynamicIndexAccessNode(it)) {}

DynamicIndexIterator DynamicIndexAccess::getIterator() const{
   return  getNode(*this)->it;
}

template <> bool isa<DynamicIndexAccess>(DynamicExpr e) {
  return isa<DynamicIndexAccessNode>(e.ptr);
}

template <> DynamicIndexAccess to<DynamicIndexAccess>(DynamicExpr e) {
  taco_iassert(isa<DynamicIndexAccess>(e));
  return DynamicIndexAccess(to<DynamicIndexAccessNode>(e.ptr));
}

DynamicIndexMulInternal::DynamicIndexMulInternal() :  DynamicIndexMulInternal(new DynamicIndexMulInternalNode) {}
DynamicIndexMulInternal::DynamicIndexMulInternal(const DynamicIndexMulInternalNode* n) : DynamicExpr(n){}
DynamicIndexMulInternal::DynamicIndexMulInternal(DynamicOrder dynamicOrder) : DynamicExpr(new DynamicIndexMulInternalNode(dynamicOrder)) {}

DynamicOrder DynamicIndexMulInternal::getDynamicOrder() const{
   return  getNode(*this)->dynamicOrder;
}

template <> bool isa<DynamicIndexMulInternal>(DynamicExpr e) {
  return isa<DynamicIndexMulInternalNode>(e.ptr);
}

template <> DynamicIndexMulInternal to<DynamicIndexMulInternal>(DynamicExpr e) {
  taco_iassert(isa<DynamicIndexMulInternal>(e));
  return DynamicIndexMulInternal(to<DynamicIndexMulInternalNode>(e.ptr));
}

DynamicIndexLen::DynamicIndexLen() :  DynamicIndexLen(new DynamicIndexLenNode) {}
DynamicIndexLen::DynamicIndexLen(const DynamicIndexLenNode* n) : DynamicExpr(n){}
DynamicIndexLen::DynamicIndexLen(DynamicOrder dynamicOrder) : DynamicExpr(new DynamicIndexLenNode(dynamicOrder)) {}

DynamicOrder DynamicIndexLen::getDynamicOrder() const{
   return  getNode(*this)->dynamicOrder;
}

template <> bool isa<DynamicIndexLen>(DynamicExpr e) {
  return isa<DynamicIndexLenNode>(e.ptr);
}

template <> DynamicIndexLen to<DynamicIndexLen>(DynamicExpr e) {
  taco_iassert(isa<DynamicIndexLen>(e));
  return DynamicIndexLen(to<DynamicIndexLenNode>(e.ptr));
}


DynamicAdd::DynamicAdd() : DynamicAdd(new DynamicAddNode) {}
DynamicAdd::DynamicAdd(const DynamicAddNode* n) : DynamicExpr(n){}
DynamicAdd::DynamicAdd(DynamicExpr a, DynamicExpr b) : DynamicExpr(new DynamicAddNode(a, b)) {}

DynamicExpr DynamicAdd::getA() const{
  return getNode(*this)->a;
}
DynamicExpr DynamicAdd::getB() const{
  return getNode(*this)->a;
}

template <> bool isa<DynamicAdd>(DynamicExpr e) {
  return isa<DynamicAddNode>(e.ptr);
}

template <> DynamicAdd to<DynamicAdd>(DynamicExpr e) {
  taco_iassert(isa<DynamicAdd>(e));
  return DynamicAdd(to<DynamicAddNode>(e.ptr));
}


DynamicMul::DynamicMul() : DynamicMul(new DynamicMulNode) {}
DynamicMul::DynamicMul(const DynamicMulNode* n) : DynamicExpr(n){}
DynamicMul::DynamicMul(DynamicExpr a, DynamicExpr b) : DynamicExpr(new DynamicMulNode(a, b)) {}

DynamicExpr DynamicMul::getA() const{
  return getNode(*this)->a;
}
DynamicExpr DynamicMul::getB() const{
  return getNode(*this)->a;
}

template <> bool isa<DynamicMul>(DynamicExpr e) {
  return isa<DynamicMulNode>(e.ptr);
}

template <> DynamicMul to<DynamicMul>(DynamicExpr e) {
  taco_iassert(isa<DynamicMul>(e));
  return DynamicMul(to<DynamicMulNode>(e.ptr));
}

DynamicDiv::DynamicDiv() : DynamicDiv(new DynamicDivNode) {}
DynamicDiv::DynamicDiv(const DynamicDivNode* n) : DynamicExpr(n){}
DynamicDiv::DynamicDiv(DynamicExpr a, DynamicExpr b) : DynamicExpr(new DynamicDivNode(a, b)) {}

DynamicExpr DynamicDiv::getA() const{
  return getNode(*this)->a;
}
DynamicExpr DynamicDiv::getB() const{
  return getNode(*this)->a;
}

template <> bool isa<DynamicDiv>(DynamicExpr e) {
  return isa<DynamicDivNode>(e.ptr);
}

template <> DynamicDiv to<DynamicDiv>(DynamicExpr e) {
  taco_iassert(isa<DynamicDiv>(e));
  return DynamicDiv(to<DynamicDivNode>(e.ptr));
}

DynamicMod::DynamicMod() : DynamicMod(new DynamicModNode) {}
DynamicMod::DynamicMod(const DynamicModNode* n) : DynamicExpr(n){}
DynamicMod::DynamicMod(DynamicExpr a, DynamicExpr b) : DynamicExpr(new DynamicModNode(a, b)) {}

DynamicExpr DynamicMod::getA() const{
  return getNode(*this)->a;
}
DynamicExpr DynamicMod::getB() const{
  return getNode(*this)->a;
}

template <> bool isa<DynamicMod>(DynamicExpr e) {
  return isa<DynamicModNode>(e.ptr);
}

template <> DynamicMod to<DynamicMod>(DynamicExpr e) {
  taco_iassert(isa<DynamicMod>(e));
  return DynamicMod(to<DynamicModNode>(e.ptr));
}

DynamicSub::DynamicSub() : DynamicSub(new DynamicSubNode) {}
DynamicSub::DynamicSub(const DynamicSubNode* n) : DynamicExpr(n){}
DynamicSub::DynamicSub(DynamicExpr a, DynamicExpr b) : DynamicSub(new DynamicSubNode(a, b)) {}

DynamicExpr DynamicSub::getA() const{
  return getNode(*this)->a;
}
DynamicExpr DynamicSub::getB() const{
  return getNode(*this)->a;
}

template <> bool isa<DynamicSub>(DynamicExpr e) {
  return isa<DynamicSubNode>(e.ptr);
}

template <> DynamicSub to<DynamicSub>(DynamicExpr e) {
  taco_iassert(isa<DynamicSub>(e));
  return DynamicSub(to<DynamicSubNode>(e.ptr));
}

DynamicIndexVar::DynamicIndexVar() : DynamicIndexVar(new DynamicIndexVarNode) {}
DynamicIndexVar::DynamicIndexVar(const DynamicIndexVarNode* n) : DynamicExpr(n){}
DynamicIndexVar::DynamicIndexVar(IndexVar i) : DynamicIndexVar(new DynamicIndexVarNode(i)) {}

IndexVar DynamicIndexVar::getIVar() const{
  return getNode(*this)->i;
}

template <> bool isa<DynamicIndexVar>(DynamicExpr e) {
  return isa<DynamicIndexVarNode>(e.ptr);
}

template <> DynamicIndexVar to<DynamicIndexVar>(DynamicExpr e) {
  taco_iassert(isa<DynamicIndexVar>(e));
  return DynamicIndexVar(to<DynamicIndexVarNode>(e.ptr));
}

DynamicEqual::DynamicEqual() : DynamicEqual(new DynamicEqualNode) {}
DynamicEqual::DynamicEqual(const DynamicEqualNode* n) : DynamicStmt(n){}
DynamicEqual::DynamicEqual(DynamicExpr a, DynamicExpr b) : DynamicStmt(new DynamicEqualNode(a, b)) {}

DynamicExpr DynamicEqual::getA() const{
  return getNode(*this)->a;
}
DynamicExpr DynamicEqual::getB() const{
  return getNode(*this)->a;
}

template <> bool isa<DynamicEqual>(DynamicStmt e) {
  return isa<DynamicEqualNode>(e.ptr);
}

template <> DynamicEqual to<DynamicEqual>(DynamicStmt e) {
  taco_iassert(isa<DynamicEqual>(e));
  return DynamicEqual(to<DynamicEqualNode>(e.ptr));
}

DynamicNotEqual::DynamicNotEqual() : DynamicNotEqual(new DynamicNotEqualNode) {}
DynamicNotEqual::DynamicNotEqual(const DynamicNotEqualNode* n) : DynamicStmt(n){}
DynamicNotEqual::DynamicNotEqual(DynamicExpr a, DynamicExpr b) : DynamicStmt(new DynamicNotEqualNode(a, b)) {}

DynamicExpr DynamicNotEqual::getA() const{
  return getNode(*this)->a;
}
DynamicExpr DynamicNotEqual::getB() const{
  return getNode(*this)->a;
}

template <> bool isa<DynamicNotEqual>(DynamicStmt e) {
  return isa<DynamicNotEqualNode>(e.ptr);
}

template <> DynamicNotEqual to<DynamicNotEqual>(DynamicStmt e) {
  taco_iassert(isa<DynamicNotEqual>(e));
  return DynamicNotEqual(to<DynamicNotEqualNode>(e.ptr));
}

DynamicGreater::DynamicGreater() : DynamicGreater(new DynamicGreaterNode) {}
DynamicGreater::DynamicGreater(const DynamicGreaterNode* n) : DynamicStmt(n){}
DynamicGreater::DynamicGreater(DynamicExpr a, DynamicExpr b) : DynamicStmt(new DynamicGreaterNode(a, b)) {}

DynamicExpr DynamicGreater::getA() const{
  return getNode(*this)->a;
}
DynamicExpr DynamicGreater::getB() const{
  return getNode(*this)->a;
}

template <> bool isa<DynamicGreater>(DynamicStmt e) {
  return isa<DynamicGreaterNode>(e.ptr);
}

template <> DynamicGreater to<DynamicGreater>(DynamicStmt e) {
  taco_iassert(isa<DynamicGreater>(e));
  return DynamicGreater(to<DynamicGreaterNode>(e.ptr));
}

DynamicLess::DynamicLess() : DynamicLess(new DynamicLessNode) {}
DynamicLess::DynamicLess(const DynamicLessNode* n) : DynamicStmt(n){}
DynamicLess::DynamicLess(DynamicExpr a, DynamicExpr b) : DynamicStmt(new DynamicLessNode(a, b)) {}

DynamicExpr DynamicLess::getA() const{
  return getNode(*this)->a;
}
DynamicExpr DynamicLess::getB() const{
  return getNode(*this)->a;
}

template <> bool isa<DynamicLess>(DynamicStmt e) {
  return isa<DynamicLessNode>(e.ptr);
}

template <> DynamicLess to<DynamicLess>(DynamicStmt e) {
  taco_iassert(isa<DynamicLess>(e));
  return DynamicLess(to<DynamicLessNode>(e.ptr));
}

DynamicGeq::DynamicGeq() : DynamicGeq(new DynamicGeqNode) {}
DynamicGeq::DynamicGeq(const DynamicGeqNode* n) : DynamicStmt(n){}
DynamicGeq::DynamicGeq(DynamicExpr a, DynamicExpr b) : DynamicStmt(new DynamicGeqNode(a, b)) {}

DynamicExpr DynamicGeq::getA() const{
  return getNode(*this)->a;
}
DynamicExpr DynamicGeq::getB() const{
  return getNode(*this)->a;
}

template <> bool isa<DynamicGeq>(DynamicStmt e) {
  return isa<DynamicGeqNode>(e.ptr);
}

template <> DynamicGeq to<DynamicGeq>(DynamicStmt e) {
  taco_iassert(isa<DynamicGeq>(e));
  return DynamicGeq(to<DynamicGeqNode>(e.ptr));
}

DynamicLeq::DynamicLeq() : DynamicLeq(new DynamicLeqNode) {}
DynamicLeq::DynamicLeq(const DynamicLeqNode* n) : DynamicStmt(n){}
DynamicLeq::DynamicLeq(DynamicExpr a, DynamicExpr b) : DynamicStmt(new DynamicLeqNode(a, b)) {}

DynamicExpr DynamicLeq::getA() const{
  return getNode(*this)->a;
}
DynamicExpr DynamicLeq::getB() const{
  return getNode(*this)->a;
}

template <> bool isa<DynamicLeq>(DynamicStmt e) {
  return isa<DynamicLeqNode>(e.ptr);
}

template <> DynamicLeq to<DynamicLeq>(DynamicStmt e) {
  taco_iassert(isa<DynamicLeq>(e));
  return DynamicLeq(to<DynamicLeqNode>(e.ptr));
}

DynamicForall::DynamicForall() : DynamicForall(new DynamicForallNode) {}
DynamicForall::DynamicForall(const DynamicForallNode* n) : DynamicStmt(n){}
DynamicForall::DynamicForall(DynamicIndexIterator it, DynamicStmt stmt) : DynamicStmt(new DynamicForallNode(it, stmt)) {}

DynamicIndexIterator DynamicForall::getIterator() const{
    return getNode(*this)->it;
}

DynamicStmt DynamicForall::getStmt() const{
  return getNode(*this)->stmt;
}

template <> bool isa<DynamicForall>(DynamicStmt e) {
  return isa<DynamicForallNode>(e.ptr);
}

template <> DynamicForall to<DynamicForall>(DynamicStmt e) {
  taco_iassert(isa<DynamicForall>(e));
  return DynamicForall(to<DynamicForallNode>(e.ptr));
}

DynamicExists::DynamicExists() : DynamicExists(new DynamicExistsNode) {}
DynamicExists::DynamicExists(const DynamicExistsNode* n) : DynamicStmt(n){}
DynamicExists::DynamicExists(DynamicIndexIterator it, DynamicStmt stmt) : DynamicStmt(new DynamicExistsNode(it, stmt)) {}

DynamicIndexIterator DynamicExists::getIterator() const{
    return getNode(*this)->it;
}

DynamicStmt DynamicExists::getStmt() const{
  return getNode(*this)->stmt;
}

template <> bool isa<DynamicExists>(DynamicStmt e) {
  return isa<DynamicExistsNode>(e.ptr);
}

template <> DynamicExists to<DynamicExists>(DynamicStmt e) {
  taco_iassert(isa<DynamicExists>(e));
  return DynamicExists(to<DynamicExistsNode>(e.ptr));
}

DynamicAnd::DynamicAnd() : DynamicAnd(new DynamicAndNode) {}
DynamicAnd::DynamicAnd(const DynamicAndNode* n) : DynamicStmt(n) {}
DynamicAnd::DynamicAnd(DynamicStmt a, DynamicStmt b) : DynamicAnd(new DynamicAndNode(a, b)) {}

DynamicStmt DynamicAnd::getA() const{
  return  getNode(*this)->a;
}

DynamicStmt DynamicAnd::getB() const{
  return  getNode(*this)->b;
}

template <> bool isa<DynamicAnd>(DynamicStmt e) {
  return isa<DynamicAndNode>(e.ptr);
}

template <> DynamicAnd to<DynamicAnd>(DynamicStmt e) {
  taco_iassert(isa<DynamicAnd>(e));
  return DynamicAnd(to<DynamicAndNode>(e.ptr));
}

DynamicOr::DynamicOr() : DynamicOr(new DynamicOrNode) {}
DynamicOr::DynamicOr(const DynamicOrNode* n) : DynamicStmt(n) {}
DynamicOr::DynamicOr(DynamicStmt a, DynamicStmt b) : DynamicOr(new DynamicOrNode(a, b)) {}

DynamicStmt DynamicOr::getA() const{
  return  getNode(*this)->a;
}

DynamicStmt DynamicOr::getB() const{
  return  getNode(*this)->b;
}

template <> bool isa<DynamicOr>(DynamicStmt e) {
  return isa<DynamicOrNode>(e.ptr);
}

template <> DynamicOr to<DynamicOr>(DynamicStmt e) {
  taco_iassert(isa<DynamicOr>(e));
  return DynamicOr(to<DynamicOrNode>(e.ptr));
}

PropertyExpr::PropertyExpr(std::string property) : PropertyExpr(new PropertyTagNode(property)) {}

void PropertyExpr::accept(PropertyExprVisitorStrict * v) const{
  ptr->accept(v);
}

std::ostream& operator<<(std::ostream& os, const PropertyExpr& expr) {
  if (!expr.defined()) return os << "DynamicExpr()";
  PropertyNotationPrinter printer(os);
  printer.print(expr);
  return os;
}

PropertyExpr operator+(const PropertyExpr& a, const PropertyExpr& b){
  return PropertyAdd(a, b);
}

PropertyExpr operator-(const PropertyExpr& a, const PropertyExpr& b){
  return PropertySub(a, b);
}

PropertyExpr operator*(const PropertyExpr& a, const PropertyExpr& b){
  return PropertyMul(a, b);
}

PropertyExpr operator/(const PropertyExpr& a, const PropertyExpr& b){
  return PropertyDiv(a, b);
}

PropertyTag::PropertyTag() : PropertyExpr(new PropertyTagNode) {}
PropertyTag::PropertyTag(const PropertyTagNode* n) : PropertyExpr(n) {}
PropertyTag::PropertyTag(const std::string& tag) : PropertyTag(new PropertyTagNode(tag)) {}

std::string PropertyTag::getTag() const{
  return getNode(*this)->property;
}

template <> bool isa<PropertyTag>(PropertyExpr e) {
  return isa<PropertyTagNode>(e.ptr);
}

template <> PropertyTag to<PropertyTag>(PropertyExpr e) {
  taco_iassert(isa<PropertyTag>(e));
  return PropertyTag(to<PropertyTagNode>(e.ptr));
}

PropertyAssign PropertyTag::operator=(const PropertyExpr& expr){
  return PropertyAssign(*this, expr);
}

PropertyAssign PropertyTag::operator=(const PropertyExpr& expr) const{
   return PropertyAssign(*this, expr);
}

PropertyAdd::PropertyAdd() : PropertyExpr(new PropertyAddNode) {}
PropertyAdd::PropertyAdd(const PropertyAddNode* n) : PropertyExpr(n) {}
PropertyAdd::PropertyAdd(PropertyExpr a, PropertyExpr b) : PropertyAdd(new PropertyAddNode(a, b)) {}

PropertyExpr PropertyAdd::getA() const{
  return getNode(*this)->a;
}
PropertyExpr PropertyAdd::getB() const{
   return getNode(*this)->b;
}

template <> bool isa<PropertyAdd>(PropertyExpr e) {
  return isa<PropertyAddNode>(e.ptr);
}

template <> PropertyAdd to<PropertyAdd>(PropertyExpr e) {
  taco_iassert(isa<PropertyAdd>(e));
  return PropertyAdd(to<PropertyAddNode>(e.ptr));
}

PropertySub::PropertySub() : PropertyExpr(new PropertySubNode) {}
PropertySub::PropertySub(const PropertySubNode* n) : PropertyExpr(n) {}
PropertySub::PropertySub(PropertyExpr a, PropertyExpr b) : PropertySub(new PropertySubNode(a, b)) {}

PropertyExpr PropertySub::getA() const{
  return getNode(*this)->a;
}
PropertyExpr PropertySub::getB() const{
   return getNode(*this)->b;
}

template <> bool isa<PropertySub>(PropertyExpr e) {
  return isa<PropertySubNode>(e.ptr);
}

template <> PropertySub to<PropertySub>(PropertyExpr e) {
  taco_iassert(isa<PropertySub>(e));
  return PropertySub(to<PropertySubNode>(e.ptr));
}

PropertyMul::PropertyMul() : PropertyExpr(new PropertyMulNode) {}
PropertyMul::PropertyMul(const PropertyMulNode* n) : PropertyExpr(n) {}
PropertyMul::PropertyMul(PropertyExpr a, PropertyExpr b) : PropertyMul(new PropertyMulNode(a, b)) {}

PropertyExpr PropertyMul::getA() const{
  return getNode(*this)->a;
}
PropertyExpr PropertyMul::getB() const{
   return getNode(*this)->b;
}

template <> bool isa<PropertyMul>(PropertyExpr e) {
  return isa<PropertyMulNode>(e.ptr);
}

template <> PropertyMul to<PropertyMul>(PropertyExpr e) {
  taco_iassert(isa<PropertyMul>(e));
  return PropertyMul(to<PropertyMulNode>(e.ptr));
}

PropertyDiv::PropertyDiv() : PropertyExpr(new PropertyDivNode) {}
PropertyDiv::PropertyDiv(const PropertyDivNode* n) : PropertyExpr(n) {}
PropertyDiv::PropertyDiv(PropertyExpr a, PropertyExpr b) : PropertyDiv(new PropertyDivNode(a, b)) {}

PropertyExpr PropertyDiv::getA() const{
  return getNode(*this)->a;
}
PropertyExpr PropertyDiv::getB() const{
   return getNode(*this)->b;
}

template <> bool isa<PropertyDiv>(PropertyExpr e) {
  return isa<PropertyDivNode>(e.ptr);
}

template <> PropertyDiv to<PropertyDiv>(PropertyExpr e) {
  taco_iassert(isa<PropertyDiv>(e));
  return PropertyDiv(to<PropertyDivNode>(e.ptr));
}

PropertyAssign::PropertyAssign() : PropertyStmt(new PropertyAssignNode) {}
PropertyAssign::PropertyAssign(const PropertyAssignNode* n) : PropertyStmt(n) {}
PropertyAssign::PropertyAssign(PropertyTag rhs, PropertyExpr lhs) : PropertyAssign(new PropertyAssignNode(rhs, lhs)) {}

PropertyTag PropertyAssign::getLhs() const{
   return getNode(*this)->lhs;
}

PropertyExpr PropertyAssign::getRhs() const{
   return getNode(*this)->rhs;
}

template <> bool isa<PropertyAssign>(PropertyStmt e) {
  return isa<PropertyAssignNode>(e.ptr);
}

template <> PropertyAssign to<PropertyAssign>(PropertyStmt e) {
  taco_iassert(isa<PropertyAssign>(e));
  return PropertyAssign(to<PropertyAssignNode>(e.ptr));
}

void PropertyStmt::accept(PropertyStmtVisitorStrict * v) const{
  ptr->accept(v);
}

std::ostream& operator<<(std::ostream& os, const PropertyStmt& expr) {
  if (!expr.defined()) return os << "DynamicStmt()";
  PropertyNotationPrinter printer(os);
  printer.print(expr);
  return os;
}


}