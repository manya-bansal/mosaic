#include "taco/index_notation/accel_interface.h"

using namespace std; 

namespace taco {


std::ostream& operator<<(std::ostream& os, const TransferWithArgs& transferWithArgs){

  os << transferWithArgs.returnType << " " << transferWithArgs.name << "(" << util::join(transferWithArgs.args) << ")" << std::endl;

  return os; 
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

}

