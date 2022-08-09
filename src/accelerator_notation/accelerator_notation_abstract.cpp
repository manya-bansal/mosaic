#include "taco/accelerator_notation/accelerator_notation_nodes_abstract.h"


using namespace std;

namespace taco {


AcceleratorExprNode::AcceleratorExprNode(Datatype type) : dataType(type) {}

Datatype AcceleratorExprNode::getDataType() const {
  return dataType;
}

AcceleratorStmtNode::AcceleratorStmtNode(Type type) : type (type) {}


Type AcceleratorStmtNode::getType() const {
    return type;
}

}