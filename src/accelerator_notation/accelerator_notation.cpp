#include "taco/index_notation/index_notation.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <utility>
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
#include "taco/accelerator_notation/accelerator_notation_nodes.h"
#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/accelerator_notation/accelerator_notation_printer.h"
#include "taco/accelerator_notation/accelerate_search.h"
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

/// Must override the default Access operator=, otherwise it is a copy.
AcceleratorAssignment AcceleratorAccess::operator=(const AcceleratorAccess& access){
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

//class AcceleratorNegNode
AcceleratorNeg::AcceleratorNeg(const AcceleratorNegNode* n) : AcceleratorExpr(n) {
}

AcceleratorNeg::AcceleratorNeg(AcceleratorExpr a) : AcceleratorNeg(new AcceleratorNegNode(a)) {
};

AcceleratorExpr AcceleratorNeg::getA() const {
  return getNode(*this)->a;
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

//class AcceleratorSqrt
AcceleratorSqrt::AcceleratorSqrt(const AcceleratorSqrtNode* n) : AcceleratorExpr(n) {
}

AcceleratorSqrt::AcceleratorSqrt(AcceleratorExpr a) : AcceleratorSqrt(new AcceleratorSqrtNode(a)) {
};

AcceleratorExpr AcceleratorSqrt::getA() const {
  return getNode(*this)->a;
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

//class AcceleratorAssigment
AcceleratorAssignment::AcceleratorAssignment(const AcceleratorAssignmentNode* n) : AcceleratorStmt(n) {
}

AcceleratorAssignment::AcceleratorAssignment(AcceleratorAccess lhs, AcceleratorExpr rhs, AcceleratorExpr op)
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
};

/// Sets the name of the tensor object.
void TensorObject::setName(const std::string& name) const{
  content->name = name;
}

/// Returns the type of the tensor variable.
const Type& TensorObject::getType() const{
  return content->type;
}

int TensorObject::getOrder() const {
  return content->type.getShape().getOrder();
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

AcceleratorAssignment TensorObject::operator=(AcceleratorExpr expr) {
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


}