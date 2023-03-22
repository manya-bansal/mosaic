#include "taco/accelerator_notation/accelerator_notation_printer.h"
#include "taco/accelerator_notation/accelerator_notation_nodes.h"
#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/accelerator_notation/code_gen_dynamic_order.h"
#include "taco/util/env.h"
#include <fstream>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <chrono>


using namespace std;

namespace taco {

GenerateSMTCode::GenerateSMTCode(const DynamicStmt& stmtLower, const std::map<DynamicOrder, std::vector<IndexVar>>& dynamicOrderToVar, 
    const std::map<IndexVar, int>& varToDim, bool tile) 
: stmtLower(stmtLower), dynamicOrderToVar(dynamicOrderToVar), varToDim(varToDim), tile(tile) {

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

static std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

//from https://stackoverflow.com/questions/9435385/split-a-string-using-c11
std::vector<std::string> split(const std::string &s, char delim) {
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> elems;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

static std::map<IndexVar, int> compare(std::map<IndexVar, int> a, std::map<IndexVar, int> b){
    int aProduct = 1;
    for (auto val: a){
        aProduct *= val.second;
    }

    int bProduct = 1;
    for (auto val: b){
        bProduct *= val.second;
    }

    if (aProduct > bProduct){
        return a; 
    }

    return b;
}

std::map<IndexVar, int> GenerateSMTCode::getTilings(){
    std::map<IndexVar, int> tilings;
    if (!isSat()){
        return tilings;
    }
    std::string SMTRetrun = runSMT();
    std::vector<std::string> potentialTilings = split(SMTRetrun, '\n');
    bool first = true; 
    for (auto potentialTiling : potentialTilings){
        std::map<IndexVar, int> curTiling;
        if (first){
            first = false; 
            continue;
        }
        potentialTiling = potentialTiling.substr(1, potentialTiling.size()-2);
        std::vector<std::string> indexVarTilings = split(potentialTiling, ',');
        for (auto tiling : indexVarTilings){
            tiling.erase(remove(tiling.begin(), tiling.end(), ' '), tiling.end());
            std::vector<std::string> value = split(tiling, '=');
            taco_uassert(nameToVar.count(value[0]));
            curTiling[nameToVar[value[0]]] = stoi(value[1]); 
        }

        if (tilings.size() == 0){
            tilings = curTiling;
        }else{
            tilings = compare(tilings, curTiling);
        }
    }
    return tilings;
}

bool GenerateSMTCode::isSat(){
    std::string result = runSMT();
    if (result.substr(0,3) == "sat"){
        return true;
    }
    return false; 

}

// GROSS!!!!!
std::string GenerateSMTCode::runSMT(){
    auto start = std::chrono::high_resolution_clock::now();
    std::string pythonCode = generatePythonCode();
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by function: "
         << duration.count() << " us" << std::endl;
    //gets generated in build/bin
    ofstream SMTPython("SMTpython.py");
    SMTPython << pythonCode;
    SMTPython.close();
    start = std::chrono::high_resolution_clock::now();
    std::string result =  exec("python3 SMTpython.py");
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // system("rm -rf SMTpython.py");
    return result; 
}

std::string GenerateSMTCode::generatePythonCode(){
    //declare var
    string pythonCode;
    pythonCode += "import z3\n\n"
                  "s = z3.Solver()\n";

    std::set<std::string> emitted;
    for (auto entry : indexVarName){
        if (emitted.count(entry.second)){
            continue;
        }
        pythonCode += entry.second + " = z3.Int(\'" + entry.second + "\')\n";
        pythonCode += "s.add(" + entry.second + " > 0)\n";
        emitted.insert(entry.second);
        taco_uassert(varToDim.count(entry.first));
        if (tile){
                pythonCode += "s.add(" + entry.second + " < " + std::to_string(varToDim[entry.first]) + ")\n";
            }else{
                pythonCode += "s.add(" +  entry.second + " == " + std::to_string(varToDim[entry.first]) + ")\n";
        }
        nameToVar[entry.second] = entry.first;
    }

    for (auto entry: dynamicOrderToVar){
        for (auto var : entry.second){
            if (emitted.count(var.getName())){
                continue;
            }
            pythonCode += var.getName() + " = z3.Int(\'" + var.getName() + "\')\n";
            pythonCode += "s.add(" + var.getName() + " > 0)\n";
            emitted.insert(var.getName());
            taco_uassert(varToDim.count(var));
            if (tile){
                pythonCode += "s.add(" + var.getName() + " < " + std::to_string(varToDim[var]) + ")\n";
            }else{
                pythonCode += "s.add(" + var.getName() + " == " + std::to_string(varToDim[var]) + ")\n";
            }

            nameToVar[var.getName()] = var;
        }
    }

    stmtLower.accept(this);
    pythonCode += "\ns.add(" + this->s +")\n\n";

    pythonCode += "print(s.check())\n";
    //now we ennumerate solutions
    pythonCode += "while s.check() == z3.sat:\n";
    std::vector<std::string> conditions; 
    //only inlcude values that are greater than the ones we have seen before 
    for (auto var : emitted){
        conditions.push_back(var + " > " + "s.model()[" + var + "]");
    }

    pythonCode += "\tprint(s.model())\n";
    pythonCode += "\ts.add(z3.Or(" + util::join(conditions) + "))\n";
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