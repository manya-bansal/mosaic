#include "taco/index_notation/accel_interface.h"

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
 };

std::ostream& TransferWithArgs::print(std::ostream& os) const{
  os << returnType << " " << name << "(" << util::join(args) << ")";
  return os;
}

void TensorPropertiesArgs::lower() const {
  std::cout << "lower with Expr" << std::endl;
};

std::ostream& TensorPropertiesArgs::print(std::ostream& os) const{
  switch(internalArgType){
    case TENSORVAR:
      os << t.getName();
      break;
    case DIM:
      os << "Dim(" << indexVar << ")";
      break;
    case EXPR:
      os << irExpr;
      break;
    default:
      os << "Unknown! Add case!";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const TensorPropertiesArgs& tensorPropertiesArgs){
  return tensorPropertiesArgs.print(os);
};


}

