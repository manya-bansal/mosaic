#include "taco/accelerator_notation/accel_interface.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/tensor.h"

using namespace std; 

namespace taco {

ArgType Argument::getArgType() const{
  if (ptr == NULL) return UNKNOWN;
  return getNode()->argType;
}

std::ostream& operator<<(std::ostream& os,  const Argument& argument){
  return argument.getNode()->print(os);
}


struct TransferType::Content {
  std::string name;
  taco::TransferLoad transferLoad;
  taco::TransferStore transferStore;
  bool isTensorVar;
  TensorVar tensorVar;
};

TransferType::TransferType(std::string name, taco::TransferLoad transferLoad, taco::TransferStore transferStore)
  : content (new TransferType::Content){

  content->name = name;
  content->transferLoad = transferLoad;
  content->transferStore = transferStore;
}

void TransferWithArgs::lower() const {
  std::cout << "lower with Transfer" << std::endl;
}

std::ostream& TransferWithArgs::print(std::ostream& os) const{
  os << returnType << " " << name << "(" << util::join(args) << ")";
  return os;
}

std::ostream& TensorVarArg::print(std::ostream& os) const{
    os << t.getName();
    return os;
}

Argument TensorVarArg::operator=(Argument func) const{
  taco_uassert(func.getArgType() == USER_DEFINED);
  auto f = func.getNode<TransferWithArgs>();
  return new TransferWithArgs(*f, new TensorVarArg(t));
}

std::ostream& TensorObjectArg::print(std::ostream& os) const{
  os << t << endl;
  return os;
}

Argument TensorObjectArg::operator=(Argument func) const{
  taco_uassert(func.getArgType() == USER_DEFINED);
  auto f = func.getNode<TransferWithArgs>();
  return new TransferWithArgs(*f, new TensorObjectArg(t));
}

std::ostream& irExprArg::print(std::ostream& os) const{
  os << irExpr;
  return os;
}

std::ostream& DimArg::print(std::ostream& os) const{
  os << "Dim(" << indexVar << ")";
  return os;
}

std::ostream& TensorArg::print(std::ostream& os) const{
  os << irExpr;
  return os;
}

std::ostream& LiteralArg::print(std::ostream& os) const{
  

  switch (datatype.getKind()){
    case Datatype::UInt32:
      os << this->getVal<int32_t>();
      break;
    default: 
      os << "Add CASE!";
  }
  return os;
}


Argument DeclVar::operator=(Argument func) const{
  taco_uassert(func.getArgType() == USER_DEFINED);
  auto t = func.getNode<TransferWithArgs>();
  return new TransferWithArgs(*t, new DeclVarArg(*this));
}


std::ostream& operator<<(std::ostream& os, const ForeignFunctionDescription& foreignFunctionDescription){
  os << foreignFunctionDescription.returnType <<  " " << foreignFunctionDescription.functionName << "(" 
    << util::join(foreignFunctionDescription.args) << ")";
  return os;
}

// void ConcreteAccelerateCodeGenerator::operator=(const ConcreteAccelerateCodeGenerator& concreteAccelerateCodeGenerator)
// ConcreteAccelerateCodeGenerator& ConcreteAccelerateCodeGenerator::operator=(const ConcreteAccelerateCodeGenerator& concreteAccelerateCodeGenerator) {
//       std::cout << concreteAccelerateCodeGenerator.rhs << std::endl;
//       std::cout << concreteAccelerateCodeGenerator << std::endl;
//       std::cout << *this <<std::endl;
//       return *this;
//       // return ConcreteAccelerateCodeGenerator(concreteAccelerateCodeGenerator.functionName, concreteAccelerateCodeGenerator.returnType, concreteAccelerateCodeGenerator.lhs,
//                                               // concreteAccelerateCodeGenerator.rhs, concreteAccelerateCodeGenerator.args, concreteAccelerateCodeGenerator.declarations);
// }

std::ostream& operator<<(std::ostream& os,  const ConcreteAccelerateCodeGenerator& accelGen){
  os << accelGen.getReturnType() << " " << accelGen.getFunctionName() << "(" << util::join(accelGen.getArguments()) 
    << ") targets " << accelGen.getLHS() << " = " << accelGen.getRHS() << endl;
  return os;
}

  


}

