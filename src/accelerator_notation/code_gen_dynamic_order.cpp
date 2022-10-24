#include "taco/accelerator_notation/accelerator_notation_printer.h"
#include "taco/accelerator_notation/accelerator_notation_nodes.h"
#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/accelerator_notation/code_gen_dynamic_order.h"

using namespace std;

namespace taco {

GenerateSMTCode::GenerateSMTCode(const DynamicStmt& stmtLower, const std::map<DynamicOrder, std::vector<IndexVar>>& dynamicOrderToVar) 
: stmtLower(stmtLower), dynamicOrderToVar(dynamicOrderToVar) {

    struct Visitor : DynamicNotationVisitor {
        using DynamicNotationVisitor::visit;
        map<IndexVar, string> indexVarName;
        void constructIndexVarNames(DynamicStmt stmtLower) {
            if (!stmtLower.defined()) return;
            stmtLower.accept(this);
        }
        void visit(const DynamicIndexVarNode* op) {
            indexVarName[op->i] = op->i.getName();
        }
    };

    Visitor visitStmt; 
    visitStmt.constructIndexVarNames(stmtLower);
    indexVarName = visitStmt.indexVarName;
}

std::string GenerateSMTCode::generatePythonCode(){
    //declare var
    stmtLower.accept(this);
    string pythonCode = "s.add(" + this->s +")\n";
    //ennumerate solutions
    return pythonCode;
}

std::string GenerateSMTCode::lower(DynamicStmt stmt){
    stmt.accept(this);
    return this->s;
}
std::string GenerateSMTCode::lower(DynamicExpr expr){
    expr.accept(this);
    return this->s;
}

void GenerateSMTCode::visit(const DynamicIndexIteratorNode* op){
    taco_uassert(curIterator.count(DynamicIndexIterator(op)));
    s =  std::to_string(curIterator[DynamicIndexIterator(op)]);
}
void GenerateSMTCode::visit(const DynamicIndexAccessNode* op){
    taco_uassert(curIterator.count(op->it));
    taco_uassert(dynamicOrderToVar.count(op->it.getDynamicOrder()));
    std::vector<IndexVar> indices = dynamicOrderToVar[op->it.getDynamicOrder()];
    s = indices[curIterator[op->it]].getName();
}
void GenerateSMTCode::visit(const DynamicLiteralNode* op){
    s = std::to_string(op->num);
}
void GenerateSMTCode::visit(const DynamicIndexLenNode* op){
    taco_uassert(dynamicOrderToVar.count(op->dynamicOrder));
    s = std::to_string(dynamicOrderToVar[op->dynamicOrder].size());
}
void GenerateSMTCode::visit(const DynamicIndexMulInternalNode* op){
    taco_uassert(dynamicOrderToVar.count(op->dynamicOrder));
    taco_uerror << "Unimplimented";
}
void GenerateSMTCode::visit(const DynamicAddNode* op){
    s = lower(op->a) + " + " + lower(op->b);
}
void GenerateSMTCode::visit(const DynamicSubNode* op){
    s = lower(op->a) + " - " + lower(op->b);
}
void GenerateSMTCode::visit(const DynamicMulNode* op){
    s = lower(op->a) + " * " + lower(op->b);
}
void GenerateSMTCode::visit(const DynamicDivNode* op){
    s = lower(op->a) + " / " + lower(op->b);
}
void GenerateSMTCode::visit(const DynamicModNode* op){
    s = lower(op->a) + " % " + lower(op->b);
}
void GenerateSMTCode::visit(const DynamicIndexVarNode* op){
    taco_uassert(indexVarName.count(op->i));
    s = indexVarName[op->i];
}
void GenerateSMTCode::visit(const DynamicEqualNode* op){
    s = "(" + lower(op->a) + ")" + " == " + "(" + lower(op->b) + ")";
}
void GenerateSMTCode::visit(const DynamicNotEqualNode* op){
    s = "(" + lower(op->a) + ")" + " != " + "(" + lower(op->b) + ")";
}
void GenerateSMTCode::visit(const DynamicGreaterNode* op){
     s = "(" + lower(op->a) + ")" + " > " + "(" + lower(op->b) + ")";
}
void GenerateSMTCode::visit(const DynamicLessNode* op){
    s = "(" + lower(op->a) + ")" + " < " + "(" + lower(op->b) + ")";
}
void GenerateSMTCode::visit(const DynamicLeqNode* op){
    s = "(" + lower(op->a) + ")" + " <= " + "(" + lower(op->b) + ")";
}
void GenerateSMTCode::visit(const DynamicGeqNode* op){
    s = "(" + lower(op->a) + ")" + " >= " + "(" + lower(op->b) + ")";
}
void GenerateSMTCode::visit(const DynamicForallNode* op){
    std::vector<std::string> conditions; 
    //first we initialize our iterator
    if (curIterator.count(op->it)){
        taco_uerror << "Using the same iterator twice";
    }
    curIterator[op->it] = 0;
    //next we iterate
    std::vector<IndexVar> indexVars = dynamicOrderToVar[op->it.getDynamicOrder()];
    for (size_t i = 0; i < indexVars.size(); i++){
        conditions.push_back(lower(op->stmt));
        curIterator[op->it]++;
    }
    s = "z3.And(" + util::join(conditions) + ")";
    //remove entry from dict
    curIterator.erase(op->it);
}

void GenerateSMTCode::visit(const DynamicExistsNode* op){
    std::vector<std::string> conditions; 
    //first we initialize our iterator
    if (curIterator.count(op->it)){
        taco_uerror << "Using the same iterator twice";
    }
    curIterator[op->it] = 0;
    //next we iterate
    std::vector<IndexVar> indexVars = dynamicOrderToVar[op->it.getDynamicOrder()];
    for (size_t i = 0; i < indexVars.size(); i++){
        conditions.push_back(lower(op->stmt));
        curIterator[op->it]++;
    }
    s = "z3.Or(" + util::join(conditions) + ")";
    //remove entry from dict
    curIterator.erase(op->it);
}

void GenerateSMTCode::visit(const DynamicAndNode* op){
    s = "z3.And(" + lower(op->a) + "," + lower(op->b) + ")";
}

void GenerateSMTCode::visit(const DynamicOrNode* op){
    s = "z3.Or(" + lower(op->a) + "," + lower(op->b) + ")";

}

}