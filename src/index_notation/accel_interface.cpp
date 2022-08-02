#include "taco/index_notation/accel_interface.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/tensor.h"

using namespace std; 

namespace taco {


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



std::ostream& operator<<(std::ostream& os, const ForeignFunctionDescription& foreignFunctionDescription){
  os << foreignFunctionDescription.returnType <<  " " << foreignFunctionDescription.functionName << "(" 
    << util::join(foreignFunctionDescription.args) << ")";
  return os;
}


//should always have one assign
//TODO: ADD SOME ERROR CHECKING FOR THIS
taco::IndexExpr ConcreteAccelerateCodeGenerator::getExpr() {

  return rhs;
} 

std::ostream& operator<<(std::ostream& os,  const ConcreteAccelerateCodeGenerator& accelGen){
  os << accelGen.returnType << " " << accelGen.functionName << "(" << util::join(accelGen.args) 
    << ") targets " << accelGen.lhs << " = " << accelGen.rhs << endl;
  return os;
}

  


}

