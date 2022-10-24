#ifndef CODE_GEN_DYNAMIC_ORDER_H
#define CODE_GEN_DYNAMIC_ORDER_H

#include <ostream>
#include "taco/accelerator_notation/accelerator_notation_visitor.h"

namespace taco {

class GenerateSMTCode : public DynamicNotationVisitorStrict {
public:
  GenerateSMTCode(const DynamicStmt& stmtLower, const std::map<DynamicOrder, std::vector<IndexVar>>& dynamicOrderToVar);

  std::string generatePythonCode();
  std::string lower(DynamicStmt stmt);
  std::string lower(DynamicExpr expr);

  

  using DynamicNotationVisitorStrict::visit;

  void visit(const DynamicIndexIteratorNode*);
  void visit(const DynamicIndexAccessNode*);
  void visit(const DynamicLiteralNode*);
  void visit(const DynamicIndexLenNode*);
  void visit(const DynamicIndexMulInternalNode*);
  void visit(const DynamicAddNode*);
  void visit(const DynamicSubNode*);
  void visit(const DynamicMulNode*);
  void visit(const DynamicDivNode*);
  void visit(const DynamicModNode*);
  void visit(const DynamicIndexVarNode*);

  void visit(const DynamicEqualNode*);
  void visit(const DynamicNotEqualNode*);
  void visit(const DynamicGreaterNode*);
  void visit(const DynamicLessNode*);
  void visit(const DynamicLeqNode*);
  void visit(const DynamicGeqNode*);
  void visit(const DynamicForallNode*);
  void visit(const DynamicExistsNode*);
  void visit(const DynamicAndNode*);
  void visit(const DynamicOrNode*);

private:
  DynamicStmt stmtLower;
  std::map<DynamicOrder, std::vector<IndexVar>> dynamicOrderToVar;
  std::string s;
  std::map<IndexVar, std::string> indexVarName;
};

}

#endif