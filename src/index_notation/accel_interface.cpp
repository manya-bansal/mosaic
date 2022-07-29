#include "taco/index_notation/accel_interface.h"

using namespace std; 

namespace taco {


// std::ostream& operator<<(std::ostream& os, const TransferWithArgs& transferWithArgs){

//   os << transferWithArgs.getReturnType() << " " << transferWithArgs.getName() << "(" << util::join(transferWithArgs.getArgs()) << ")" << std::endl;

//   return os; 
// }

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
  os << returnType << " " << name << "(" << util::join(args) << ") ";
  return os;
}

void TensorPropertiesArgs::lower() const {
  std::cout << "lower with Expr" << std::endl;
};


std::ostream& TensorPropertiesArgs::print(std::ostream& os) const{
  os << irExpr;
  return os;
}

// std::ostream& operator<<(std::ostream& os, const TransferTypeArgs& transferTypeArgs){
//   os << "hello" << std::endl;
// };

std::ostream& operator<<(std::ostream& os, const TensorPropertiesArgs& tensorPropertiesArgs){
  os << tensorPropertiesArgs.irExpr << std::endl;
};

// struct ForeignFunctionDescription::Content {
//   taco::IndexStmt targetStmt;
//   std::string functionName;
//   std::string returnType;
//   std::vector<TransferTypeArgs> args;
//   std::vector<taco::TensorVar> temporaries;
//   std::function<bool(taco::IndexStmt)> checker;
//   //TODO: some way to define config ??
  
// };

// ForeignFunctionDescription::ForeignFunctionDescription(const IndexStmt& targetStmt, const std::string& functionName, const std::string& returnType, const std::vector<TransferTypeArgs>& args, 
//     const std::vector<TensorVar>& temporaries, std::function<bool(taco::IndexStmt)> checker) : content (new ForeignFunctionDescription::Content) {

//       content->targetStmt = targetStmt;
//       content->functionName = functionName;
//       content->returnType = returnType;
//       // content->args = args;
//       // content->temporaries = temporaries;
//       content->checker = checker;
//     }


}

