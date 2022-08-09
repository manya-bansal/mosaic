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
#include "taco/index_notation/accel_interface.h"
#include "taco/index_notation/transformations.h"
#include "taco/accelerator_notation/accelerator_notation_nodes.h"
#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/accelerator_notation/accelerator_notation_printer.h"
#include "taco/index_notation/accelerate_search.h"
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

Datatype AcceleratorExpr::getDataType() const {
  return const_cast<AcceleratorExprNode*>(this->ptr)->getDataType();
}

std::ostream& operator<<(std::ostream& os, const AcceleratorExpr& expr) {
  if (!expr.defined()) return os << "AcceleratorExpr()";
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


}