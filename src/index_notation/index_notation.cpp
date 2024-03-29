#include "taco/index_notation/index_notation.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <utility>
#include <set>
#include <stack>
#include <tuple>
#include <taco/ir/simplify.h>
#include "lower/mode_access.h"

#include "error/error_checks.h"
#include "taco/error/error_messages.h"
#include "taco/type.h"
#include "taco/format.h"

#include "taco/index_notation/properties.h"
#include "taco/index_notation/intrinsic.h"
#include "taco/index_notation/schedule.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/index_notation/transformations.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation_printer.h"
#include "taco/accelerator_notation/accelerate_search.h"
#include "taco/accelerator_notation/accelerator_notation_nodes.h"
#include "taco/accelerator_notation/code_gen_dynamic_order.h"
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

// class IndexExpr
IndexExpr::IndexExpr(TensorVar var) 
    : IndexExpr(new AccessNode(var,{},{},false)) {
}

IndexExpr::IndexExpr(char val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(int8_t val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(int16_t val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(int32_t val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(int64_t val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(uint8_t val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(uint16_t val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(uint32_t val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(uint64_t val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(float val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(double val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(std::complex<float> val) :IndexExpr(new LiteralNode(val)){
}

IndexExpr::IndexExpr(std::complex<double> val) :IndexExpr(new LiteralNode(val)){
}

Datatype IndexExpr::getDataType() const {
  return const_cast<IndexExprNode*>(this->ptr)->getDataType();
}

std::vector<IndexVar> IndexExpr::getIndexVars(){

vector<IndexVar> vars;;
  set<IndexVar> seen;
  match(*this,
    std::function<void(const AssignmentNode*,Matcher*)>([&](
        const AssignmentNode* op, Matcher* ctx) {
      for (auto& var : op->lhs.getIndexVars()) {
        if (!util::contains(seen, var)) {
          vars.push_back(var);
          seen.insert(var);
        }
      }
      ctx->match(op->rhs);
    }),
    std::function<void(const AccessNode*)>([&](const AccessNode* op) {
      for (auto& var : op->indexVars) {
        if (!util::contains(seen, var)) {
          vars.push_back(var);
          seen.insert(var);
        }
      }
    })
  );
  return vars;
}

std::map<IndexVar,Dimension> IndexExpr::getIndexVarDomains() const{
  map<IndexVar, Dimension> indexVarDomains;
  match(*this,
    function<void(const AccessNode*)>([&indexVarDomains](const AccessNode* op) {
      auto& type = op->tensorVar.getType();
      auto& vars = op->indexVars;
      for (size_t i = 0; i < vars.size(); i++) {
        if (!util::contains(indexVarDomains, vars[i])) {
          indexVarDomains.insert({vars[i], type.getShape().getDimension(i)});
        }
        else {
          taco_iassert(indexVarDomains.at(vars[i]) ==
                       type.getShape().getDimension(i))
              << "Index variable used to index incompatible dimensions";
        }
      }
    })
  );

  return indexVarDomains;

}

Shape IndexExpr::getShape() const {

  std::vector<Dimension> indexVarDims;
  std::set<IndexVar> seenVars;

  // cout << this << endl;

  std::vector<const AccessNode*> readNodes = error::getAccessNodes(*this);
  for (auto& readNode : readNodes) {
    for (size_t mode = 0; mode < readNode->indexVars.size(); mode++) {
      IndexVar var = readNode->indexVars[mode];
      Dimension dimension = readNode->tensorVar.getType().getShape().getDimension(mode);

      // If this access has windowed modes, use the dimensions of those windows
      // as the shape, rather than the shape of the underlying tensor.
      auto a = Access(readNode);
      if (a.isModeWindowed(mode)) {
        dimension = Dimension(a.getWindowSize(mode));
      } else if (a.isModeIndexSet(mode)) {
        dimension = Dimension(a.getIndexSet(mode).size());
      }

      if (seenVars.find(var) == seenVars.end()){
         indexVarDims.push_back(dimension);
         seenVars.insert(var);
      }

    }
  }

  return Shape(indexVarDims);

}

void IndexExpr::workspace(IndexVar i, IndexVar iw, std::string name) {
//  const_cast<IndexExprNode*>(this->ptr)->splitOperator(i, i, iw);
}

void IndexExpr::workspace(IndexVar i, IndexVar iw, Format format, string name) {
//  const_cast<IndexExprNode*>(this->ptr)->splitOperator(i, i, iw);
}

void IndexExpr::workspace(IndexVar i, IndexVar iw, TensorVar workspace) {
//  const_cast<IndexExprNode*>(this->ptr)->splitOperator(i, i, iw);
//  const_cast<IndexExprNode*>(this->ptr)->workspace(i, iw, workspace);
  this->ptr->setWorkspace(i, iw, workspace);
}

void IndexExpr::accept(IndexExprVisitorStrict *v) const {
  ptr->accept(v);
}

std::ostream& operator<<(std::ostream& os, const IndexExpr& expr) {
  if (!expr.defined()) return os << "IndexExpr()";
  IndexNotationPrinter printer(os);
  printer.print(expr);
  return os;
}

static bool checkRegionDefinitions(const CallNode* anode, const CallNode* bnode) {
  // Check region definitions
  if (anode->regionDefinitions.size() != bnode->regionDefinitions.size()) {
    return false;
  }

  auto& aDefs = anode->regionDefinitions;
  auto& bDefs = bnode->regionDefinitions;
  for (auto itA = aDefs.begin(), itB = bDefs.begin(); itA != aDefs.end(); ++itA, ++itB) {
    if(itA->first != itB->first) {
      return false;
    }

    std::vector<IndexExpr> aArgs;
    std::vector<IndexExpr> bArgs;
    for(int idx : itA->first) {
      taco_iassert((size_t)idx < anode->args.size()); // We already know anode->args.size == bnode->args.size
      aArgs.push_back(anode->args[idx]);
      bArgs.push_back(bnode->args[idx]);
    }

    // TODO lower and check IR
    if(!util::targetPtrEqual(itA->second, itB->second)) {
      return false;
    }
  }

  return true;
}

/// Checks if the iteration algebra structure is the same and the ordering of the index expressions
/// nested under regions is the same for each op node.
static bool checkIterationAlg(const CallNode* anode, const CallNode* bnode) {
  // Check IterationAlgebra structures
  if(!algStructureEqual(anode->iterAlg, bnode->iterAlg)) {
    return false;
  }

  struct OrderChecker : public IterationAlgebraVisitor {
    explicit OrderChecker(const CallNode* op) : op(op) {}

    std::vector<size_t>& check() {
      op->iterAlg.accept(this);
      return ordering;
    }

    using IterationAlgebraVisitor::visit;

    void visit(const RegionNode* region) {
      const IndexExpr& e = region->expr();
      auto it = std::find(op->args.begin(), op->args.end(), e);
      taco_iassert(it != op->args.end()) << "Iteration algebra region expressions must be in arguments";
      size_t loc = it - op->args.begin();
      ordering.push_back(loc);
    }

    std::vector<size_t> ordering;
    const CallNode* op;
  };

  std::vector<size_t> aOrdering = OrderChecker(anode).check();
  std::vector<size_t> bOrdering = OrderChecker(bnode).check();
  return aOrdering == bOrdering;
}

struct Isomorphic : public IndexNotationVisitorStrict {
  bool eq = false;
  IndexExpr bExpr;
  IndexStmt bStmt;
  std::map<TensorVar,TensorVar> isoATensor, isoBTensor;
  std::map<IndexVar,IndexVar> isoAVar, isoBVar;

  bool check(IndexExpr a, IndexExpr b) {
    if (!a.defined() && !b.defined()) {
      return true;
    }
    if ((a.defined() && !b.defined()) || (!a.defined() && b.defined())) {
      return false;
    }
    this->bExpr = b;
    a.accept(this);
    return eq;
  }

  bool check(IndexStmt a, IndexStmt b) {
    if (!a.defined() && !b.defined()) {
      return true;
    }
    if ((a.defined() && !b.defined()) || (!a.defined() && b.defined())) {
      return false;
    }
    this->bStmt = b;
    a.accept(this);
    return eq;
  }

  bool check(TensorVar a, TensorVar b) {
    if (!util::contains(isoBTensor, a) && !util::contains(isoATensor, b)) {
      if (a.getType() != b.getType() || a.getFormat() != b.getFormat()) {
        return false;
      }
      isoBTensor.insert({a, b});
      isoATensor.insert({b, a});
      return true;
    }
    if (!util::contains(isoBTensor, a) || !util::contains(isoATensor, b)) {
      return false;
    }
    return (isoBTensor[a] == b) && (isoATensor[b] == a);
  }

  bool check(IndexVar a, IndexVar b) {
    if (!util::contains(isoBVar, a) && !util::contains(isoAVar, b)) {
      isoBVar.insert({a, b});
      isoAVar.insert({b, a});
      return true;
    }
    if (!util::contains(isoBVar, a) || !util::contains(isoAVar, b)) {
      return false;
    }
    return (isoBVar[a] == b) && (isoAVar[b] == a);
  }

  using IndexNotationVisitorStrict::visit;

  void visit(const IndexVarNode* anode) {
    if(!isa<IndexVarNode>(bExpr.ptr)) {
      eq = false;
      return;
    }

    auto bnode = to<IndexVarNode>(bExpr.ptr);
    if(anode != bnode) {
      eq = false;
      return;
    }

    eq = true;
  }

  void visit(const AccessNode* anode) {
    if (!isa<AccessNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<AccessNode>(bExpr.ptr);
    if (!check(anode->tensorVar, bnode->tensorVar)) {
      eq = false;
      return;
    }
    if (anode->indexVars.size() != bnode->indexVars.size()) {
      eq = false;
      return;
    }
    for (size_t i = 0; i < anode->indexVars.size(); i++) {
      if (!check(anode->indexVars[i], bnode->indexVars[i])) {
        eq = false;
        return;
      }
    }
    if (anode->isAccessingStructure != bnode->isAccessingStructure ||
        anode->windowedModes != bnode->windowedModes) {
      eq = false;
      return;
    }
    if (anode->indexSetModes.size() != bnode->indexSetModes.size()) {
      eq = false;
      return;
    }
    for (auto aset = anode->indexSetModes.begin(), bset = bnode->indexSetModes.begin(); aset != anode->indexSetModes.end(); ++aset, ++bset) {
      if (aset->first != bset->first || *aset->second.set != *bset->second.set) {
        eq = false;
        return;
      }
    }
    eq = true;
  }

  void visit(const LiteralNode* anode) {
    if (!isa<LiteralNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<LiteralNode>(bExpr.ptr);
    if (anode->getDataType() != bnode->getDataType()) {
      eq = false;
      return;
    }
    if (memcmp(anode->val,bnode->val,anode->getDataType().getNumBytes()) != 0) {
      eq = false;
      return;
    }
    eq = true;
  }

  template <class T>
  bool unaryIsomorphic(const T* anode, IndexExpr b) {
    if (!isa<T>(b.ptr)) {
      return false;
    }
    auto bnode = to<T>(b.ptr);
    if (!check(anode->a, bnode->a)) {
      return false;
    }
    return true;
  }

  void visit(const NegNode* anode) {
    eq = unaryIsomorphic(anode, bExpr);
  }

  void visit(const SqrtNode* anode) {
    eq = unaryIsomorphic(anode, bExpr);
  }

  template <class T>
  bool binaryIsomorphic(const T* anode, IndexExpr b) {
    if (!isa<T>(b.ptr)) {
      return false;
    }
    auto bnode = to<T>(b.ptr);
    if (!check(anode->a, bnode->a) || !check(anode->b, bnode->b)) {
      return false;
    }
    return true;
  }

  void visit(const AddNode* anode) {
    eq = binaryIsomorphic(anode, bExpr);
  }

  void visit(const SubNode* anode) {
    eq = binaryIsomorphic(anode, bExpr);
  }

  void visit(const MulNode* anode) {
    eq = binaryIsomorphic(anode, bExpr);
  }

  void visit(const DivNode* anode) {
    eq = binaryIsomorphic(anode, bExpr);
  }

  void visit(const CastNode* anode) {
    if (!isa<CastNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<CastNode>(bExpr.ptr);
    if (anode->getDataType() != bnode->getDataType() ||
        !check(anode->a, bnode->a)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const CallIntrinsicNode* anode) {
    if (!isa<CallIntrinsicNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<CallIntrinsicNode>(bExpr.ptr);
    if (anode->func->getName() != bnode->func->getName() ||
        anode->args.size() != bnode->args.size()) {
      eq = false;
      return;
    }
    for (size_t i = 0; i < anode->args.size(); ++i) {
      if (!check(anode->args[i], bnode->args[i])) {
        eq = false;
        return;
      }
    }
    eq = true;
  }

  void visit(const ReductionNode* anode) {
    if (!isa<ReductionNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<ReductionNode>(bExpr.ptr);
    if (!check(anode->op, bnode->op) ||
        !check(anode->var, bnode->var) ||
        !check(anode->a, bnode->a)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const AssignmentNode* anode) {
    if (!isa<AssignmentNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<AssignmentNode>(bStmt.ptr);
    if (!check(anode->lhs, bnode->lhs) ||
        !check(anode->rhs, bnode->rhs) ||
        !check(anode->op, bnode->op)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const YieldNode* anode) {
    if (!isa<YieldNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<YieldNode>(bStmt.ptr);
    if (anode->indexVars.size() != bnode->indexVars.size()) {
      eq = false;
      return;
    }
    for (size_t i = 0; i < anode->indexVars.size(); i++) {
      if (!check(anode->indexVars[i], bnode->indexVars[i])) {
        eq = false;
        return;
      }
    }
    if (!check(anode->expr, bnode->expr)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const ForallNode* anode) {
    if (!isa<ForallNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<ForallNode>(bStmt.ptr);
    if (!check(anode->indexVar, bnode->indexVar) ||
        !check(anode->stmt, bnode->stmt) ||
        anode->parallel_unit != bnode->parallel_unit ||
        anode->output_race_strategy != bnode->output_race_strategy ||
        anode->unrollFactor != bnode->unrollFactor) {
      eq = false;
      return;
    }
    eq = true;
  }


  void visit(const ForallManyNode* anode) {
    if (!isa<ForallManyNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<ForallManyNode>(bStmt.ptr);

    if (anode->stmts.size() != bnode->stmts.size()){
      eq = false;
      return;
    }

    for (size_t i = 0; i < anode->stmts.size(); i++){
      if (!check(anode->stmts[i], bnode->stmts[i])){
         eq = false;
        return;
      }
    }

    eq = true;
  }

  void visit(const WhereNode* anode) {
    if (!isa<WhereNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<WhereNode>(bStmt.ptr);
    if (!check(anode->consumer, bnode->consumer) ||
        !check(anode->producer, bnode->producer)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const DimReductionNode* anode) {
    if (!isa<DimReductionNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<DimReductionNode>(bStmt.ptr);
    if (!check(anode->consumer, bnode->consumer) ||
        !check(anode->producer, bnode->producer)) {
      eq = false;
      return;
    }
    eq = true;
  }

    void visit(const AccelerateNode* anode) {
      if (!isa<AccelerateNode>(bStmt.ptr)) {
        eq = false;
        return;
      }
      auto bnode = to<AccelerateNode>(bStmt.ptr);
      if (!check(anode->consumer, bnode->consumer) ||
          !check(anode->producer, bnode->producer)) {
        eq = false;
        return;
      }
      eq = true;
  }

  void visit(const InterfaceCallNode* anode) {
      if (!isa<InterfaceCallNode>(bStmt.ptr)) {
        eq = false;
        return;
      }
      auto bnode = to<InterfaceCallNode>(bStmt.ptr);
      if (!check(anode->producer, bnode->producer)) {
        eq = false;
        return;
      }
      eq = true;
  }

  void visit(const SequenceNode* anode) {
    if (!isa<SequenceNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<SequenceNode>(bStmt.ptr);
    if (!check(anode->definition, bnode->definition) ||
        !check(anode->mutation, bnode->mutation)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const AssembleNode* anode) {
    if (!isa<AssembleNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<AssembleNode>(bStmt.ptr);
    if (!check(anode->queries, bnode->queries) ||
        !check(anode->compute, bnode->compute)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const MultiNode* anode) {
    if (!isa<MultiNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<MultiNode>(bStmt.ptr);
    if (!check(anode->stmt1, bnode->stmt1) ||
        !check(anode->stmt2, bnode->stmt2)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const SuchThatNode* anode) {
    if (!isa<SuchThatNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<SuchThatNode>(bStmt.ptr);
    if (!check(anode->stmt, bnode->stmt) ||
         anode->predicate != bnode->predicate) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const CallNode* anode) {
    if (!isa<CallNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<CallNode>(bExpr.ptr);

    // Properties
    if (anode->properties.size() != bnode->properties.size()) {
      eq = false;
      return;
    }

    for(const auto& a_prop : anode->properties) {
      bool found = false;
      for(const auto& b_prop : bnode->properties) {
        if(a_prop.equals(b_prop)) {
          found = true;
          break;
        }
      }
      if (!found) {
        eq = false;
        return;
      }
    }

    // Exhausted regions
    if (anode->definedRegions != bnode->definedRegions) {
      eq = false;
      return;
    }

    // Lower function
    // TODO: For now just check that the function pointers are the same.
    // TODO (rawnh): This check is broken. The retrieved function pointers are null
    //  when attempting to dereference them. The original code attempted to use
    //  util::targetPtrEqual.
    if (util::getFromEnv("TACO_ISOMORPHIC_HACK", "0") == "0") {
      if (&anode->defaultLowerFunc != &bnode->defaultLowerFunc) {
        eq = false;
        return;
      }
    } else {
      // If the hack is enabled, check that names are the same.
      if (anode->name != bnode->name) {
        eq = false;
        return;
      }
    }

    // Check arguments
    if (anode->args.size() != bnode->args.size()) {
      eq = false;
      return;
    }

    for (size_t i = 0; i < anode->args.size(); ++i) {
      if (!check(anode->args[i], bnode->args[i])) {
        eq = false;
        return;
      }
    }

    // Algebra
    if (!checkIterationAlg(anode, bnode)) {
      eq = false;
      return;
    }

    // Special definitions
    eq = checkRegionDefinitions(anode, bnode);
  }
};

bool isomorphic(IndexExpr a, IndexExpr b) {
  if (!a.defined() && !b.defined()) {
    return true;
  }
  if ((a.defined() && !b.defined()) || (!a.defined() && b.defined())) {
    return false;
  }
  return Isomorphic().check(a,b);
}

bool isomorphic(IndexStmt a, IndexStmt b) {
  if (!a.defined() && !b.defined()) {
    return true;
  }
  if ((a.defined() && !b.defined()) || (!a.defined() && b.defined())) {
    return false;
  }
  return Isomorphic().check(a,b);
}

static void addCommutativityRewrite(IndexStmt stmt, std::map<IndexExpr, std::vector<IndexExpr>> &exprToreplace){
  
  
  match(stmt,                                            
   std::function<void(const AddNode*,Matcher*)>([&](const AddNode* op,
                                                      Matcher* ctx) {
      // a + b = b + a                                                 
      exprToreplace[op].push_back(new AddNode(op->b, op->a));   

      ctx->match(op->a); 
      ctx->match(op->b);

    }),
    std::function<void(const MulNode*,Matcher*)>([&](const MulNode* op,
                                                     Matcher* ctx) {

       // a * b = b * a 

      //any order should be fine since indexVars would ensure
      //equivalence (we are not changing any index vars)    

      // if (isa<LiteralNode>((op->a).ptr) || isa<LiteralNode>((op->b).ptr)){
      exprToreplace[op].push_back(new MulNode(op->b, op->a));   
      // }

      ctx->match(op->a); 
      ctx->match(op->b);

    })
    
    );
}

static void addIdentityRewrite(IndexStmt stmt, std::map<IndexExpr, std::vector<IndexExpr>> &exprToreplace){


  // a = a + 0
  // a = a * 1
  // a = a - 0

  match(stmt,                                            
   std::function<void(const AccessNode*,Matcher*)>([&](const AccessNode* op,
                                                     Matcher* ctx) {  

      exprToreplace[op].push_back(new AddNode(op, IndexExpr(0)));   
      exprToreplace[op].push_back(new MulNode(op, IndexExpr(1))); 
      exprToreplace[op].push_back(new SubNode(op, IndexExpr(0)));  

    }),

    std::function<void(const LiteralNode*,Matcher*)>([&](const LiteralNode* op,
                                                     Matcher* ctx) {

      exprToreplace[op].push_back(new AddNode(op, IndexExpr(0)));   
      exprToreplace[op].push_back(new MulNode(op, IndexExpr(1))); 
      exprToreplace[op].push_back(new SubNode(op, IndexExpr(0)));  

    })

    );

}

static void addDistributivityRewrites(IndexStmt stmt, std::map<IndexExpr, std::vector<IndexExpr>> &exprToreplace){


  match(stmt, 
    std::function<void(const MulNode*,Matcher*)>([&](const MulNode* op,
                                                     Matcher* ctx) {
      // (a+b)*c = a*c + b*c                                                 
      if (isa<AddNode>((op->a).ptr)){
        const AddNode * addNode = to<AddNode>((op->a).ptr);
        exprToreplace[op].push_back(new AddNode(new MulNode(addNode->a, op->b), new MulNode(addNode->b, op->b)));
      }else if (isa<AddNode>((op->b).ptr)){
        const AddNode * addNode = to<AddNode>((op->b).ptr);
        exprToreplace[op].push_back(new AddNode(new MulNode(addNode->a, op->a), new MulNode(addNode->b, op->a)));
      }

      // (a-b)*c = a*c - b*c  
      if (isa<SubNode>((op->a).ptr)){
        const SubNode * subNode = to<SubNode>((op->a).ptr);
        exprToreplace[op].push_back(new SubNode(new MulNode(subNode->a, op->b), new MulNode(subNode->b, op->b)));
      }else if (isa<SubNode>((op->b).ptr)){
        const SubNode * subNode = to<SubNode>((op->b).ptr);
        exprToreplace[op].push_back(new SubNode(new MulNode(subNode->a, op->a), new MulNode(subNode->b, op->a)));
      }

      // (a/b)*c = a*c/b 
      if (isa<DivNode>((op->a).ptr)){
        const DivNode * divNode = to<DivNode>((op->a).ptr);
        exprToreplace[op].push_back(new DivNode(new MulNode(divNode->a, op->b), divNode->b));
      }else if (isa<DivNode>((op->b).ptr)){
        const DivNode * divNode = to<DivNode>((op->b).ptr);
        exprToreplace[op].push_back(new DivNode(new MulNode(divNode->a, op->a), divNode->b));
      }

      ctx->match(op->a); 
      ctx->match(op->b);

    }),
    std::function<void(const DivNode*,Matcher*)>([&](const DivNode* op,
                                                     Matcher* ctx) {

      // (a+b)/c = a/c + b/c                                                 
      if (isa<AddNode>((op->a).ptr)){
        const AddNode * addNode = to<AddNode>((op->a).ptr);
        exprToreplace[op].push_back(new AddNode(new DivNode(addNode->a, op->b), new DivNode(addNode->b, op->b)));
      }else if (isa<AddNode>((op->b).ptr)){
        const AddNode * addNode = to<AddNode>((op->b).ptr);
        exprToreplace[op].push_back(new AddNode(new DivNode(addNode->a, op->a), new DivNode(addNode->b, op->a)));
      }

      // (a-b)/c = a/c - b/c
      if (isa<SubNode>((op->a).ptr)){
        const SubNode * subNode = to<SubNode>((op->a).ptr);
        exprToreplace[op].push_back(new SubNode(new DivNode(subNode->a, op->b), new DivNode(subNode->b, op->b)));
      }else if (isa<SubNode>((op->b).ptr)){
        const SubNode * subNode = to<SubNode>((op->b).ptr);
        exprToreplace[op].push_back(new SubNode(new DivNode(subNode->a, op->a), new DivNode(subNode->b, op->a)));
      }

      // (ab)/c = a/c*b

      if (isa<MulNode>((op->a).ptr)){
        const MulNode * mulNode = to<MulNode>((op->a).ptr);
        //( ab)/c = b/c*a should be expressed by applying a commutativity rewrite and then
        // this rewrite 
        exprToreplace[op].push_back(new MulNode(new DivNode(mulNode->a, op->b), mulNode->b));
      }if (isa<MulNode>((op->b).ptr)){
        const MulNode * mulNode = to<MulNode>((op->b).ptr);
        exprToreplace[op].push_back(new MulNode(new DivNode(mulNode->a, op->a), mulNode->b));
      }


      ctx->match(op->a); 
      ctx->match(op->b);

    })

  );

}


static void takeCommonTermsOut(IndexStmt stmt, std::map<IndexExpr, std::vector<IndexExpr>> &exprToreplace){

  match(stmt,                                            
   std::function<void(const AddNode*,Matcher*)>([&](const AddNode* op,
                                                     Matcher* ctx) {  
      // a*b + a*c = a(b+c)                                                 
      if (isa<MulNode>((op->a).ptr) && isa<MulNode>((op->b).ptr)) {
         const MulNode * mulNodeA = to<MulNode>((op->a).ptr);
         const MulNode * mulNodeB = to<MulNode>((op->b).ptr);

        if (equals(mulNodeA->a, mulNodeB->a)){
          exprToreplace[op].push_back(new MulNode(mulNodeA->a, new AddNode(mulNodeA->b, mulNodeB->b)));
        }

        if (equals(mulNodeA->b, mulNodeB->a)){
          exprToreplace[op].push_back(new MulNode(mulNodeA->b, new AddNode(mulNodeA->a, mulNodeB->b)));
        }

        if (equals(mulNodeA->a, mulNodeB->b)){
          exprToreplace[op].push_back(new MulNode(mulNodeA->a, new AddNode(mulNodeA->b, mulNodeB->a)));
        }

        if (equals(mulNodeA->b, mulNodeB->b)){
          exprToreplace[op].push_back(new MulNode(mulNodeA->b, new AddNode(mulNodeA->a, mulNodeB->a)));
        }
      
      }

      if (isa<DivNode>((op->a).ptr) && isa<DivNode>((op->b).ptr)) {
        const DivNode * divNodeA = to<DivNode>((op->a).ptr);
        const DivNode * divNodeB = to<DivNode>((op->b).ptr);

        // for division only a/5 + b/5 = (a+b)/5 is the valid choice 
        if (equals(divNodeA->b, divNodeB->b)){
          exprToreplace[op].push_back(new DivNode(new AddNode(divNodeA->a, divNodeB->a), divNodeA->b));
        }
      }

    }),
    // a*b - a*c = a(b-c)           
    std::function<void(const SubNode*,Matcher*)>([&](const SubNode* op,
                                                     Matcher* ctx) {  
      // a*b + a*c = a(b+c)                                                 
      if (isa<MulNode>((op->a).ptr) && isa<MulNode>((op->b).ptr)) {
         const MulNode * mulNodeA = to<MulNode>((op->a).ptr);
         const MulNode * mulNodeB = to<MulNode>((op->b).ptr);

        if (equals(mulNodeA->a, mulNodeB->a)){
          exprToreplace[op].push_back(new MulNode(mulNodeA->a, new SubNode(mulNodeA->b, mulNodeB->b)));
        }

        if (equals(mulNodeA->b, mulNodeB->a)){
          exprToreplace[op].push_back(new MulNode(mulNodeA->b, new SubNode(mulNodeA->a, mulNodeB->b)));
        }

        if (equals(mulNodeA->a, mulNodeB->b)){
          exprToreplace[op].push_back(new MulNode(mulNodeA->a, new SubNode(mulNodeA->b, mulNodeB->a)));
        }

        if (equals(mulNodeA->b, mulNodeB->b)){
          exprToreplace[op].push_back(new MulNode(mulNodeA->b, new SubNode(mulNodeA->a, mulNodeB->a)));
        }
      
      }

      if (isa<DivNode>((op->a).ptr) && isa<DivNode>((op->b).ptr)) {
        const DivNode * divNodeA = to<DivNode>((op->a).ptr);
        const DivNode * divNodeB = to<DivNode>((op->b).ptr);

        // for division only a/5 - b/5 = (a-b)/5 is the valid choice 
        if (equals(divNodeA->b, divNodeB->b)){
          exprToreplace[op].push_back(new DivNode(new SubNode(divNodeA->a, divNodeB->a), divNodeA->b));
        }
      }

    })
  
  );

}

static void simplifyNegatives(IndexStmt stmt, std::map<IndexExpr, std::vector<IndexExpr>> &exprToreplace){

  match(stmt,                                            
   std::function<void(const AddNode*,Matcher*)>([&](const AddNode* op,
                                                     Matcher* ctx) {  

      if (isa<NegNode>((op->a).ptr)){
         exprToreplace[op].push_back(new SubNode(op->a, op->a));
      }

    }),
    std::function<void(const SubNode*,Matcher*)>([&](const SubNode* op,
                                                     Matcher* ctx) {  

      if (isa<SubNode>((op->a).ptr)){
         exprToreplace[op].push_back(new AddNode(op->a, op->a));
      }
    })

    );
}

struct Equals : public IndexNotationVisitorStrict {
  bool eq = false;
  IndexExpr bExpr;
  IndexStmt bStmt;

  bool check(IndexExpr a, IndexExpr b) {
    this->bExpr = b;
    a.accept(this);
    return eq;
  }

  bool check(IndexStmt a, IndexStmt b) {
    this->bStmt = b;
    a.accept(this);
    return eq;
  }

  using IndexNotationVisitorStrict::visit;

  void visit(const IndexVarNode* anode) {
    if(!isa<IndexVarNode>(bExpr.ptr)) {
      eq = false;
      return;
    }

    auto bnode = to<IndexVarNode>(bExpr.ptr);
    if(anode != bnode) {
      eq = false;
      return;
    }

    eq = true;
  }

  void visit(const AccessNode* anode) {
    if (!isa<AccessNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<AccessNode>(bExpr.ptr);
    if (anode->tensorVar != bnode->tensorVar) {
      eq = false;
      return;
    }
    if (anode->indexVars.size() != bnode->indexVars.size()) {
      eq = false;
      return;
    }
    for (size_t i = 0; i < anode->indexVars.size(); i++) {
      if (anode->indexVars[i] != bnode->indexVars[i]) {
        eq = false;
        return;
      }
    }
    if (anode->isAccessingStructure != bnode->isAccessingStructure ||
        anode->windowedModes != bnode->windowedModes ||
        anode->indexSetModes != bnode->indexSetModes) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const LiteralNode* anode) {
    if (!isa<LiteralNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<LiteralNode>(bExpr.ptr);
    if (anode->getDataType() != bnode->getDataType()) {
      eq = false;
      return;
    }
    if (memcmp(anode->val,bnode->val,anode->getDataType().getNumBytes()) != 0) {
      eq = false;
      return;
    }
    eq = true;
  }

  template <class T>
  bool unaryEquals(const T* anode, IndexExpr b) {
    if (!isa<T>(b.ptr)) {
      return false;
    }
    auto bnode = to<T>(b.ptr);
    if (!equals(anode->a, bnode->a)) {
      return false;
    }
    return true;
  }

  void visit(const NegNode* anode) {
    eq = unaryEquals(anode, bExpr);
  }

  void visit(const SqrtNode* anode) {
    eq = unaryEquals(anode, bExpr);
  }

  template <class T>
  bool binaryEquals(const T* anode, IndexExpr b) {
    if (!isa<T>(b.ptr)) {
      return false;
    }
    auto bnode = to<T>(b.ptr);
    if (!equals(anode->a, bnode->a) || !equals(anode->b, bnode->b)) {
      return false;
    }
    return true;
  }

  void visit(const AddNode* anode) {
    eq = binaryEquals(anode, bExpr);
  }

  void visit(const SubNode* anode) {
    eq = binaryEquals(anode, bExpr);
  }

  void visit(const MulNode* anode) {
    eq = binaryEquals(anode, bExpr);
  }

  void visit(const DivNode* anode) {
    eq = binaryEquals(anode, bExpr);
  }

  void visit(const CastNode* anode) {
    if (!isa<CastNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<CastNode>(bExpr.ptr);
    if (anode->getDataType() != bnode->getDataType() ||
        !equals(anode->a, bnode->a)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const CallNode* anode) {
    if (!isa<CallNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<CallNode>(bExpr.ptr);

    // Properties
    if (anode->properties.size() != bnode->properties.size()) {
      eq = false;
      return;
    }

    for(const auto& a_prop : anode->properties) {
      bool found = false;
      for(const auto& b_prop : bnode->properties) {
        if(a_prop.equals(b_prop)) {
          found = true;
          break;
        }
      }
      if (!found) {
        eq = false;
        return;
      }
    }

    // Exhausted regions
    if (anode->definedRegions != bnode->definedRegions) {
      eq = false;
      return;
    }

    // Lower function
    // TODO: For now just check that the function pointers are the same.
    if(!util::targetPtrEqual(anode->defaultLowerFunc, bnode->defaultLowerFunc)) {
      eq = false;
      return;
    }

    // Check arguments
    if (anode->args.size() != bnode->args.size()) {
      eq = false;
      return;
    }

    for (size_t i = 0; i < anode->args.size(); ++i) {
      if (!equals(anode->args[i], bnode->args[i])) {
        eq = false;
        return;
      }
    }

    // Algebra
    if (!checkIterationAlg(anode, bnode)) {
      eq = false;
      return;
    }

    // Special definitions
    eq = checkRegionDefinitions(anode, bnode);
  }

  void visit(const CallIntrinsicNode* anode) {
    if (!isa<CallIntrinsicNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<CallIntrinsicNode>(bExpr.ptr);
    if (anode->func->getName() != bnode->func->getName() ||
        anode->args.size() != bnode->args.size()) {
      eq = false;
      return;
    }
    for (size_t i = 0; i < anode->args.size(); ++i) {
      if (!equals(anode->args[i], bnode->args[i])) {
        eq = false;
        return;
      }
    }
    eq = true;
  }

  void visit(const ReductionNode* anode) {
    if (!isa<ReductionNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<ReductionNode>(bExpr.ptr);
    if (!equals(anode->op, bnode->op) ||
        anode->var != bnode->var ||
        !equals(anode->a, bnode->a)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const AssignmentNode* anode) {
    if (!isa<AssignmentNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<AssignmentNode>(bStmt.ptr);
    if (!equals(anode->lhs, bnode->lhs) || !equals(anode->rhs, bnode->rhs) ||
        !equals(anode->op, bnode->op)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const YieldNode* anode) {
    if (!isa<YieldNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<YieldNode>(bStmt.ptr);
    if (anode->indexVars.size() != bnode->indexVars.size()) {
      eq = false;
      return;
    }
    for (size_t i = 0; i < anode->indexVars.size(); i++) {
      if (anode->indexVars[i] != bnode->indexVars[i]) {
        eq = false;
        return;
      }
    }
    if (!equals(anode->expr, bnode->expr)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const ForallNode* anode) {
    if (!isa<ForallNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<ForallNode>(bStmt.ptr);
    if (anode->indexVar != bnode->indexVar ||
        !equals(anode->stmt, bnode->stmt) ||
        anode->parallel_unit != bnode->parallel_unit ||
        anode->output_race_strategy != bnode->output_race_strategy ||
        anode->unrollFactor != bnode->unrollFactor) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const ForallManyNode* anode) {
    if (!isa<ForallManyNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<ForallManyNode>(bStmt.ptr);

    if (anode->stmts.size() != bnode->stmts.size()){
      eq = false;
      return;
    }

    for (size_t i = 0; i < anode->stmts.size(); i++){
      if (!equals(anode->stmts[i], bnode->stmts[i])){
         eq = false;
        return;
      }
    }

    eq = true;
  }

  void visit(const WhereNode* anode) {
    if (!isa<WhereNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<WhereNode>(bStmt.ptr);
    if (!equals(anode->consumer, bnode->consumer) ||
        !equals(anode->producer, bnode->producer)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const DimReductionNode* anode) {
    if (!isa<DimReductionNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<DimReductionNode>(bStmt.ptr);
    if (!equals(anode->consumer, bnode->consumer) ||
        !equals(anode->producer, bnode->producer)) {
      eq = false;
      return;
    }
    eq = true;
  }


  void visit(const AccelerateNode* anode) {
    if (!isa<AccelerateNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<AccelerateNode>(bStmt.ptr);
    if (!equals(anode->consumer, bnode->consumer) ||
        !equals(anode->producer, bnode->producer)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const InterfaceCallNode* anode) {
    if (!isa<InterfaceCallNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<InterfaceCallNode>(bStmt.ptr);
    if (!equals(anode->producer, bnode->producer)) {
      eq = false;
      return;
    }
    eq = true;
  }


  void visit(const SequenceNode* anode) {
    if (!isa<SequenceNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<SequenceNode>(bStmt.ptr);
    if (!equals(anode->definition, bnode->definition) ||
        !equals(anode->mutation, bnode->mutation)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const AssembleNode* anode) {
    if (!isa<AssembleNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<AssembleNode>(bStmt.ptr);
    if (!equals(anode->queries, bnode->queries) ||
        !equals(anode->compute, bnode->compute)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const MultiNode* anode) {
    if (!isa<MultiNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<MultiNode>(bStmt.ptr);
    if (!equals(anode->stmt1, bnode->stmt1) ||
        !equals(anode->stmt2, bnode->stmt2)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const SuchThatNode* anode) {
    if (!isa<SuchThatNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<SuchThatNode>(bStmt.ptr);
    if (anode->predicate != bnode->predicate ||
        !equals(anode->stmt, bnode->stmt)) {
      eq = false;
      return;
    }
    eq = true;
  }
};

bool equals(IndexExpr a, IndexExpr b) {
  if (!a.defined() && !b.defined()) {
    return true;
  }
  if ((a.defined() && !b.defined()) || (!a.defined() && b.defined())) {
    return false;
  }
  return Equals().check(a,b);
}

bool equals(IndexStmt a, IndexStmt b) {
  if (!a.defined() && !b.defined()) {
    return true;
  }
  if ((a.defined() && !b.defined()) || (!a.defined() && b.defined())) {
    return false;
  }
  return Equals().check(a,b);
}

IndexExpr operator-(const IndexExpr& expr) {
  return new NegNode(expr.ptr);
}

IndexExpr operator+(const IndexExpr& lhs, const IndexExpr& rhs) {
  return new AddNode(lhs, rhs);
}

IndexExpr operator-(const IndexExpr& lhs, const IndexExpr& rhs) {
  return new SubNode(lhs, rhs);
}

IndexExpr operator*(const IndexExpr& lhs, const IndexExpr& rhs) {
  return new MulNode(lhs, rhs);
}

IndexExpr operator/(const IndexExpr& lhs, const IndexExpr& rhs) {
  return new DivNode(lhs, rhs);
}


// class Access
Access::Access(const AccessNode* n) : IndexExpr(n) {
}

Access::Access(const TensorVar& tensor, const std::vector<IndexVar>& indices,
               const std::map<int, std::shared_ptr<IndexVarIterationModifier>>& modifiers,
               bool isAccessingStructure)
    : Access(new AccessNode(tensor, indices, modifiers, isAccessingStructure)) {
}

const TensorVar& Access::getTensorVar() const {
  return getNode(*this)->tensorVar;
}

const std::vector<IndexVar>& Access::getIndexVars() const {
  return getNode(*this)->indexVars;
}

bool Access::isAccessingStructure() const {
  return getNode(*this)->isAccessingStructure;
}

bool Access::hasWindowedModes() const {
  return !getNode(*this)->windowedModes.empty();
}

bool Access::isModeWindowed(int mode) const {
  auto node = getNode(*this);
  return node->windowedModes.find(mode) != node->windowedModes.end();
}

int Access::getWindowLowerBound(int mode) const {
  taco_iassert(this->isModeWindowed(mode));
  return getNode(*this)->windowedModes.at(mode).lo;
}

int Access::getWindowUpperBound(int mode) const {
  taco_iassert(this->isModeWindowed(mode));
  return getNode(*this)->windowedModes.at(mode).hi;
}

int Access::getWindowSize(int mode) const {
  taco_iassert(this->isModeWindowed(mode));
  auto w = getNode(*this)->windowedModes.at(mode);
  return (w.hi - w.lo) / w.stride;
}

int Access::getStride(int mode) const {
  taco_iassert(this->isModeWindowed(mode));
  return getNode(*this)->windowedModes.at(mode).stride;
}

bool operator==(const Access& a, const Access& b) {
  // Short-circuit for when the Access pointers are the same.
  if (getNode(a) == getNode(b)) {
    return true;
  }
  if (a.getTensorVar() != b.getTensorVar()) {
    return false;
  }
  if (a.getIndexVars() != b.getIndexVars()) {
    return false;
  }
  if (getNode(a)->windowedModes != getNode(b)->windowedModes) {
    return false;
  }
  if (getNode(a)->indexSetModes != getNode(b)->indexSetModes) {
    return false;
  }
  return true;
}

bool operator<(const Access& a, const Access& b) {
  // First branch on tensorVar.
  if (a.getTensorVar() != b.getTensorVar()) {
    return a.getTensorVar() < b.getTensorVar();
  }

  // Then branch on the indexVars used in the access.
  if (a.getIndexVars() != b.getIndexVars()) {
    return a.getIndexVars() < b.getIndexVars();
  }

  // Branch on the windows.
  if (getNode(a)->windowedModes < getNode(b)->windowedModes) {
    return getNode(a)->windowedModes < getNode(b)->windowedModes;
  }

  // Finally, branch on the index set.
  return getNode(a)->indexSetModes < getNode(b)->indexSetModes;
}

bool Access::hasIndexSetModes() const {
  return !getNode(*this)->indexSetModes.empty();
}

bool Access::isModeIndexSet(int mode) const {
  auto node = getNode(*this);
  return util::contains(node->indexSetModes, mode);
}

TensorVar Access::getModeIndexSetTensor(int mode) const {
  taco_iassert(this->isModeIndexSet(mode));
  return getNode(*this)->indexSetModes.at(mode).tensor.getTensorVar();
}

const std::vector<int>& Access::getIndexSet(int mode) const {
  taco_iassert(this->isModeIndexSet(mode));
  return *getNode(*this)->indexSetModes.at(mode).set;
}

static void check(Assignment assignment) {
  auto lhs = assignment.getLhs();
  auto tensorVar = lhs.getTensorVar();
  auto freeVars = lhs.getIndexVars();
  auto indexExpr = assignment.getRhs();
  auto shape = tensorVar.getType().getShape();

  // If the LHS access has any windowed modes, use the dimensions of those
  // windows as the shape, rather than the shape of the underlying tensor.
  if (lhs.hasWindowedModes() || lhs.hasIndexSetModes()) {
    vector<Dimension> dims(shape.getOrder());
    for (int i = 0; i < shape.getOrder();i++) {
      dims[i] = shape.getDimension(i);
      if (lhs.isModeWindowed(i)) {
        dims[i] = Dimension(lhs.getWindowSize(i));
      } else if (lhs.isModeIndexSet(i)) {
        dims[i] = Dimension(lhs.getIndexSet(i).size());
      }
    }
    shape = Shape(dims);
  }

  auto typecheck = error::dimensionsTypecheck(freeVars, indexExpr, shape);
  taco_uassert(typecheck.first) << error::expr_dimension_mismatch << " " << typecheck.second;
}

Assignment Access::operator=(const IndexExpr& expr) {
  TensorVar result = getTensorVar();
  Assignment assignment = Assignment(*this, expr);
  check(assignment);
  const_cast<AccessNode*>(getNode(*this))->setAssignment(assignment);
  return assignment;
}

Assignment Access::operator=(const Access& expr) {
  return operator=(static_cast<IndexExpr>(expr));
}

Assignment Access::operator=(const TensorVar& var) {
  return operator=(Access(var));
}

Assignment Access::operator+=(const IndexExpr& expr) {
  TensorVar result = getTensorVar();
  Assignment assignment = Assignment(
    result,
    getIndexVars(),
    expr,
    Add(),
    // Include any windows on LHS index vars.
    getNode(*this)->packageModifiers()
  );
  // check(assignment); TODO: fix check for precompute
  const_cast<AccessNode*>(getNode(*this))->setAssignment(assignment);
  return assignment;
}

std::pair<IndexExpr, IndexExpr> Access::operator<=(IndexExpr expr){
  return std::make_pair(*this, expr);
}

template <> bool isa<Access>(IndexExpr e) {
  return isa<AccessNode>(e.ptr);
}

template <> Access to<Access>(IndexExpr e) {
  taco_iassert(isa<Access>(e));
  return Access(to<AccessNode>(e.ptr));
}


// class Literal
Literal::Literal(const LiteralNode* n) : IndexExpr(n) {
}

Literal::Literal(bool val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(unsigned char val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(unsigned short val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(unsigned int val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(unsigned long val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(unsigned long long val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(char val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(short val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(int val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(long val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(long long val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(int8_t val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(float val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(double val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(std::complex<float> val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(std::complex<double> val) : Literal(new LiteralNode(val)) {
}

Literal Literal::zero(Datatype type) {
  switch (type.getKind()) {
    case Datatype::Bool:        return Literal(false);
    case Datatype::UInt8:       return Literal(uint8_t(0));
    case Datatype::UInt16:      return Literal(uint16_t(0));
    case Datatype::UInt32:      return Literal(uint32_t(0));
    case Datatype::UInt64:      return Literal(uint64_t(0));
    case Datatype::Int8:        return Literal(int8_t(0));
    case Datatype::Int16:       return Literal(int16_t(0));
    case Datatype::Int32:       return Literal(int32_t(0));
    case Datatype::Int64:       return Literal(int64_t(0));
    case Datatype::Float32:     return Literal(float(0.0));
    case Datatype::Float64:     return Literal(double(0.0));
    case Datatype::Complex64:   return Literal(std::complex<float>());
    case Datatype::Complex128:  return Literal(std::complex<double>());
    default:                    taco_ierror << "unsupported type";
  };

  return Literal();
}

template <typename T> T Literal::getVal() const {
  return getNode(*this)->getVal<T>();
}
template bool Literal::getVal() const;
template unsigned char Literal::getVal() const;
template unsigned short Literal::getVal() const;
template unsigned int Literal::getVal() const;
template unsigned long Literal::getVal() const;
template unsigned long long Literal::getVal() const;
template char Literal::getVal() const;
template short Literal::getVal() const;
template int Literal::getVal() const;
template long Literal::getVal() const;
template long long Literal::getVal() const;
template int8_t Literal::getVal() const;
template float Literal::getVal() const;
template double Literal::getVal() const;
template std::complex<float> Literal::getVal() const;
template std::complex<double> Literal::getVal() const;

void* Literal::getValPtr() {
  return getNode(*this)->val;
}

template <> bool isa<Literal>(IndexExpr e) {
  return isa<LiteralNode>(e.ptr);
}

template <> Literal to<Literal>(IndexExpr e) {
  taco_iassert(isa<Literal>(e));
  return Literal(to<LiteralNode>(e.ptr));
}


// class Neg
Neg::Neg(const NegNode* n) : IndexExpr(n) {
}

Neg::Neg(IndexExpr a) : Neg(new NegNode(a)) {
}

IndexExpr Neg::getA() const {
  return getNode(*this)->a;
}

template <> bool isa<Neg>(IndexExpr e) {
  return isa<NegNode>(e.ptr);
}

template <> Neg to<Neg>(IndexExpr e) {
  taco_iassert(isa<Neg>(e));
  return Neg(to<NegNode>(e.ptr));
}


// class Add
Add::Add() : Add(new AddNode) {
}

Add::Add(const AddNode* n) : IndexExpr(n) {
}

Add::Add(IndexExpr a, IndexExpr b) : Add(new AddNode(a, b)) {
}

IndexExpr Add::getA() const {
  return getNode(*this)->a;
}

IndexExpr Add::getB() const {
  return getNode(*this)->b;
}

template <> bool isa<Add>(IndexExpr e) {
  return isa<AddNode>(e.ptr);
}

template <> Add to<Add>(IndexExpr e) {
  taco_iassert(isa<Add>(e));
  return Add(to<AddNode>(e.ptr));
}


// class Sub
Sub::Sub() : Sub(new SubNode) {
}

Sub::Sub(const SubNode* n) : IndexExpr(n) {
}

Sub::Sub(IndexExpr a, IndexExpr b) : Sub(new SubNode(a, b)) {
}

IndexExpr Sub::getA() const {
  return getNode(*this)->a;
}

IndexExpr Sub::getB() const {
  return getNode(*this)->b;
}

template <> bool isa<Sub>(IndexExpr e) {
  return isa<SubNode>(e.ptr);
}

template <> Sub to<Sub>(IndexExpr e) {
  taco_iassert(isa<Sub>(e));
  return Sub(to<SubNode>(e.ptr));
}


// class Mul
Mul::Mul() : Mul(new MulNode) {
}

Mul::Mul(const MulNode* n) : IndexExpr(n) {
}

Mul::Mul(IndexExpr a, IndexExpr b) : Mul(new MulNode(a, b)) {
}

IndexExpr Mul::getA() const {
  return getNode(*this)->a;
}

IndexExpr Mul::getB() const {
  return getNode(*this)->b;
}

template <> bool isa<Mul>(IndexExpr e) {
  return isa<MulNode>(e.ptr);
}

template <> Mul to<Mul>(IndexExpr e) {
  taco_iassert(isa<Mul>(e));
  return Mul(to<MulNode>(e.ptr));
}


// class Div
Div::Div() : Div(new DivNode) {
}

Div::Div(const DivNode* n) : IndexExpr(n) {
}

Div::Div(IndexExpr a, IndexExpr b) : Div(new DivNode(a, b)) {
}

IndexExpr Div::getA() const {
  return getNode(*this)->a;
}

IndexExpr Div::getB() const {
  return getNode(*this)->b;
}

template <> bool isa<Div>(IndexExpr e) {
  return isa<DivNode>(e.ptr);
}

template <> Div to<Div>(IndexExpr e) {
  taco_iassert(isa<Div>(e));
  return Div(to<DivNode>(e.ptr));
}


// class Sqrt
Sqrt::Sqrt(const SqrtNode* n) : IndexExpr(n) {
}

Sqrt::Sqrt(IndexExpr a) : Sqrt(new SqrtNode(a)) {
}

IndexExpr Sqrt::getA() const {
  return getNode(*this)->a;
}

template <> bool isa<Sqrt>(IndexExpr e) {
  return isa<SqrtNode>(e.ptr);
}

template <> Sqrt to<Sqrt>(IndexExpr e) {
  taco_iassert(isa<Sqrt>(e));
  return Sqrt(to<SqrtNode>(e.ptr));
}


// class Cast
Cast::Cast(const CastNode* n) : IndexExpr(n) {
}

Cast::Cast(IndexExpr a, Datatype newType) : Cast(new CastNode(a, newType)) {
}

IndexExpr Cast::getA() const {
  return getNode(*this)->a;
}

template <> bool isa<Cast>(IndexExpr e) {
  return isa<CastNode>(e.ptr);
}

template <> Cast to<Cast>(IndexExpr e) {
  taco_iassert(isa<Cast>(e));
  return Cast(to<CastNode>(e.ptr));
}

// class Call, most construction should happen from tensor_operator.h
Call::Call(const CallNode* n) : IndexExpr(n) {
}

Call::Call(const CallNode *n, std::string name) : IndexExpr(n), name(name) {
}

const std::vector<IndexExpr>& Call::getArgs() const {
  return getNode(*this)->args;
}

const CallNode::OpImpl Call::getFunc() const {
  return getNode(*this)->defaultLowerFunc;
}

const IterationAlgebra& Call::getAlgebra() const {
  return getNode(*this)->iterAlg;
}

const std::vector<Property>& Call::getProperties() const {
  return getNode(*this)->properties;
}

const std::string Call::getName() const {
  return getNode(*this)->name;
}

const std::map<std::vector<int>, CallNode::OpImpl> Call::getDefs() const {
  return getNode(*this)->regionDefinitions;
}

const std::vector<int>& Call::getDefinedArgs() const {
  return getNode(*this)->definedRegions;
}


template <> bool isa<Call>(IndexExpr e) {
  return isa<CallNode>(e.ptr);
}

template <> Call to<Call>(IndexExpr e) {
  taco_iassert(isa<Call>(e));
  return Call(to<CallNode>(e.ptr));
}

// class CallIntrinsic
CallIntrinsic::CallIntrinsic(const CallIntrinsicNode* n) : IndexExpr(n) {
}

CallIntrinsic::CallIntrinsic(const std::shared_ptr<Intrinsic>& func,
                             const std::vector<IndexExpr>& args)
    : CallIntrinsic(new CallIntrinsicNode(func, args)) {
}

const Intrinsic& CallIntrinsic::getFunc() const {
  return *(getNode(*this)->func);
}

const std::vector<IndexExpr>& CallIntrinsic::getArgs() const {
  return getNode(*this)->args;
}

template <> bool isa<CallIntrinsic>(IndexExpr e) {
  return isa<CallIntrinsicNode>(e.ptr);
}

template <> CallIntrinsic to<CallIntrinsic>(IndexExpr e) {
  taco_iassert(isa<CallIntrinsic>(e));
  return CallIntrinsic(to<CallIntrinsicNode>(e.ptr));
}

IndexExpr mod(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<ModIntrinsic>(), {a, b});
}

IndexExpr abs(IndexExpr a) {
  return CallIntrinsic(std::make_shared<AbsIntrinsic>(), {a});
}

IndexExpr pow(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<PowIntrinsic>(), {a, b});
}

IndexExpr square(IndexExpr a) {
  return CallIntrinsic(std::make_shared<SquareIntrinsic>(), {a});
}

IndexExpr cube(IndexExpr a) {
  return CallIntrinsic(std::make_shared<CubeIntrinsic>(), {a});
}

IndexExpr sqrt(IndexExpr a) {
  return CallIntrinsic(std::make_shared<SqrtIntrinsic>(), {a});
}

IndexExpr cbrt(IndexExpr a) {
  return CallIntrinsic(std::make_shared<CbrtIntrinsic>(), {a});
}

IndexExpr exp(IndexExpr a) {
  return CallIntrinsic(std::make_shared<ExpIntrinsic>(), {a});
}

IndexExpr log(IndexExpr a) {
  return CallIntrinsic(std::make_shared<LogIntrinsic>(), {a});
}

IndexExpr log10(IndexExpr a) {
  return CallIntrinsic(std::make_shared<Log10Intrinsic>(), {a});
}

IndexExpr sin(IndexExpr a) {
  return CallIntrinsic(std::make_shared<SinIntrinsic>(), {a});
}

IndexExpr cos(IndexExpr a) {
  return CallIntrinsic(std::make_shared<CosIntrinsic>(), {a});
}

IndexExpr tan(IndexExpr a) {
  return CallIntrinsic(std::make_shared<TanIntrinsic>(), {a});
}

IndexExpr asin(IndexExpr a) {
  return CallIntrinsic(std::make_shared<AsinIntrinsic>(), {a});
}

IndexExpr acos(IndexExpr a) {
  return CallIntrinsic(std::make_shared<AcosIntrinsic>(), {a});
}

IndexExpr atan(IndexExpr a) {
  return CallIntrinsic(std::make_shared<AtanIntrinsic>(), {a});
}

IndexExpr atan2(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<Atan2Intrinsic>(), {a, b});
}

IndexExpr sinh(IndexExpr a) {
  return CallIntrinsic(std::make_shared<SinhIntrinsic>(), {a});
}

IndexExpr cosh(IndexExpr a) {
  return CallIntrinsic(std::make_shared<CoshIntrinsic>(), {a});
}

IndexExpr tanh(IndexExpr a) {
  return CallIntrinsic(std::make_shared<TanhIntrinsic>(), {a});
}

IndexExpr asinh(IndexExpr a) {
  return CallIntrinsic(std::make_shared<AsinhIntrinsic>(), {a});
}

IndexExpr acosh(IndexExpr a) {
  return CallIntrinsic(std::make_shared<AcoshIntrinsic>(), {a});
}

IndexExpr atanh(IndexExpr a) {
  return CallIntrinsic(std::make_shared<AtanhIntrinsic>(), {a});
}

IndexExpr gt(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<GtIntrinsic>(), {a, b});
}

IndexExpr lt(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<LtIntrinsic>(), {a, b});
}

IndexExpr gte(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<GteIntrinsic>(), {a, b});
}

IndexExpr lte(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<LteIntrinsic>(), {a, b});
}

IndexExpr eq(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<EqIntrinsic>(), {a, b});
}

IndexExpr neq(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<NeqIntrinsic>(), {a, b});
}

IndexExpr max(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<MaxIntrinsic>(), {a, b});
}

IndexExpr min(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<MinIntrinsic>(), {a, b});
}

IndexExpr heaviside(IndexExpr a, IndexExpr b) {
  if (!b.defined()) {
    b = Literal::zero(a.getDataType());
  }
  return CallIntrinsic(std::make_shared<HeavisideIntrinsic>(), {a, b});
}

IndexExpr Not(IndexExpr a) {
  return CallIntrinsic(std::make_shared<NotIntrinsic>(), {a});
}


// class Reduction
Reduction::Reduction(const ReductionNode* n) : IndexExpr(n) {
}

Reduction::Reduction(IndexExpr op, IndexVar var, IndexExpr expr)
    : Reduction(new ReductionNode(op, var, expr)) {
}

IndexExpr Reduction::getOp() const {
  return getNode(*this)->op;
}

IndexVar Reduction::getVar() const {
  return getNode(*this)->var;
}

IndexExpr Reduction::getExpr() const {
  return getNode(*this)->a;
}

Reduction sum(IndexVar i, IndexExpr expr) {
  return Reduction(Add(), i, expr);
}

template <> bool isa<Reduction>(IndexExpr s) {
  return isa<ReductionNode>(s.ptr);
}

template <> Reduction to<Reduction>(IndexExpr s) {
  taco_iassert(isa<Reduction>(s));
  return Reduction(to<ReductionNode>(s.ptr));
}


// class IndexStmt
IndexStmt::IndexStmt() : util::IntrusivePtr<const IndexStmtNode>(nullptr) {
}

IndexStmt::IndexStmt(const IndexStmtNode* n)
    : util::IntrusivePtr<const IndexStmtNode>(n) {
}

void IndexStmt::accept(IndexStmtVisitorStrict *v) const {
  ptr->accept(v);
}

std::vector<IndexVar> IndexStmt::getIndexVars() const {
  vector<IndexVar> vars;;
  set<IndexVar> seen;
  match(*this,
    std::function<void(const AssignmentNode*,Matcher*)>([&](
        const AssignmentNode* op, Matcher* ctx) {
      for (auto& var : op->lhs.getIndexVars()) {
        if (!util::contains(seen, var)) {
          vars.push_back(var);
          seen.insert(var);
        }
      }
      ctx->match(op->rhs);
    }),
    std::function<void(const AccessNode*)>([&](const AccessNode* op) {
      for (auto& var : op->indexVars) {
        if (!util::contains(seen, var)) {
          vars.push_back(var);
          seen.insert(var);
        }
      }
    })
  );
  return vars;
}

map<IndexVar,Dimension> IndexStmt::getIndexVarDomains() const {
  map<IndexVar, Dimension> indexVarDomains;
  match(*this,
    std::function<void(const AssignmentNode*,Matcher*)>([](
        const AssignmentNode* op, Matcher* ctx) {
      ctx->match(op->lhs);
      ctx->match(op->rhs);
    }),
    function<void(const AccessNode*)>([&indexVarDomains](const AccessNode* op) {
      auto& type = op->tensorVar.getType();
      auto& vars = op->indexVars;
      for (size_t i = 0; i < vars.size(); i++) {
        if (!util::contains(indexVarDomains, vars[i])) {
          indexVarDomains.insert({vars[i], type.getShape().getDimension(i)});
        }
        else {
          taco_iassert(indexVarDomains.at(vars[i]) ==
                       type.getShape().getDimension(i))
              << "Index variable used to index incompatible dimensions";
        }
      }
    })
  );

  return indexVarDomains;
}

IndexStmt IndexStmt::concretizeScheduled(ProvenanceGraph provGraph, vector<IndexVar> forallIndexVarList) const {
  IndexStmt stmt = *this;
  string r;
  if (isEinsumNotation(stmt, &r)) {
    stmt = makeReductionNotationScheduled(stmt, provGraph);
  }
  if (isReductionNotationScheduled(stmt, provGraph, &r)) {
    stmt = makeConcreteNotationScheduled(stmt, provGraph, forallIndexVarList);
  }
  return stmt;
}

IndexStmt IndexStmt::concretize() const {
  IndexStmt stmt = *this;
  if (isEinsumNotation(stmt)) {
    stmt = makeReductionNotation(stmt);
  }
  if (isReductionNotation(stmt)) {
    stmt = makeConcreteNotation(stmt);
  }
  return stmt;
}

IndexStmt IndexStmt::concretizeAccelerated(const std::vector<FunctionInterface>& functionInterface) const {

  IndexStmt stmt = *this;
  if (isEinsumNotation(stmt)) {
    stmt = makeReductionNotation(stmt);
  }

  std::vector<IndexStmt> stmts =  autoAccelerate(stmt, functionInterface);
  stmt = stmts[0];

  if (isReductionNotation(stmt)) {
    stmt = makeConcreteNotation(stmt);
  }
  
  return stmt;
}

IndexStmt IndexStmt::split(IndexVar i, IndexVar i1, IndexVar i2, size_t splitFactor) const {
  IndexVarRel rel = IndexVarRel(new SplitRelNode(i, i1, i2, splitFactor));
  string reason;

  // Add predicate to concrete index notation
  IndexStmt transformed = Transformation(AddSuchThatPredicates({rel})).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  // Replace all occurrences of i with nested i1, i2
  transformed = Transformation(ForAllReplace({i}, {i1, i2})).apply(transformed, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  return transformed;
}

IndexStmt IndexStmt::splitWithoutRewrite(IndexVar i, IndexVar i1, IndexVar i2, size_t splitFactor) const {
  IndexVarRel rel = IndexVarRel(new SplitRelNode(i, i1, i2, splitFactor));
  string reason;
  // Replace all occurrences of i with nested i1, i2
  IndexStmt transformed = Transformation(AddSuchThatPredicates({rel})).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  return transformed;
}


IndexStmt IndexStmt::divide(IndexVar i, IndexVar i1, IndexVar i2, size_t splitFactor) const {
  IndexVarRel rel = IndexVarRel(new DivideRelNode(i, i1, i2, splitFactor));
  string reason;

  // Add predicate to concrete index notation.
  IndexStmt transformed = Transformation(AddSuchThatPredicates({rel})).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  // Replace all occurrences of i with nested i1, i2.
  transformed = Transformation(ForAllReplace({i}, {i1, i2})).apply(transformed, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  return transformed;
}

IndexStmt IndexStmt::precompute(IndexExpr expr, std::vector<IndexVar> i_vars,
                                std::vector<IndexVar> iw_vars, TensorVar workspace) const {

  IndexStmt transformed = *this;
  string reason;

 taco_uassert(i_vars.size() == iw_vars.size()) << "The precompute transformation requires"
                                               << "i_vars and iw_vars to be the same size";
 for (int l = 0; l < (int) i_vars.size(); l++) {
    IndexVar i = i_vars.at(l);
    IndexVar iw = iw_vars.at(l);

    if (i != iw) {
      IndexVarRel rel = IndexVarRel(new PrecomputeRelNode(i, iw));
      transformed = Transformation(AddSuchThatPredicates({rel})).apply(transformed, &reason);
      if (!transformed.defined()) {
        taco_uerror << reason;
      }
    }
  }

  transformed = Transformation(Precompute(expr, i_vars, iw_vars, workspace)).apply(transformed, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }
  return transformed;
}

IndexStmt IndexStmt::precompute(IndexExpr expr, IndexVar i, IndexVar iw, TensorVar workspace) const {
  std::vector<IndexVar> i_vars{i};
  std::vector<IndexVar> iw_vars{iw};
  return precompute(expr, i_vars, iw_vars, workspace);
}

static bool setByReference(AcceleratorStmt stmt){

  if (!isa<AcceleratorAssignment>(stmt)){
    taco_uerror << "Reference statement in function interface must be an assignemnt" << endl;
  }
  AcceleratorAssignment assign = to<AcceleratorAssignment>(stmt);

  TensorObject resultVar;
  acceleratorMatch((assign.getLhs()),
      // should only be one var on the lhs
      std::function<void(const AcceleratorAccessNode*)>([&](const AcceleratorAccessNode* op) {
            resultVar = op->tensorObject;
        })
  );

  bool setResultByRef = false;
  acceleratorMatch((assign.getRhs()),
      std::function<void(const AcceleratorAccessNode*)>([&](const AcceleratorAccessNode* op) {
            if (resultVar == op->tensorObject){
              setResultByRef = true;
            }
        })
  );
  return setResultByRef;
}

IndexStmt IndexStmt::reorder(taco::IndexVar i, taco::IndexVar j) const {
  string reason;
  IndexStmt transformed = Reorder(i, j).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }
  return transformed;
}

IndexStmt IndexStmt::reorder(std::vector<IndexVar> reorderedvars) const {
  string reason;
  IndexStmt transformed = Reorder(reorderedvars).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }
  return transformed;
}

IndexStmt IndexStmt::mergeby(IndexVar i, MergeStrategy strategy) const {
  string reason;
  IndexStmt transformed = SetMergeStrategy(i, strategy).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }
  return transformed;
}

IndexStmt IndexStmt::parallelize(IndexVar i, ParallelUnit parallel_unit, OutputRaceStrategy output_race_strategy) const {
  string reason;
  IndexStmt transformed = Parallelize(i, parallel_unit, output_race_strategy).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }
  return transformed;
}

IndexStmt IndexStmt::pos(IndexVar i, IndexVar ipos, Access access) const {
  // check access is contained in stmt
  bool foundAccess = false;
  for (Access argAccess : getArgumentAccesses(*this)) {
    if (argAccess.getTensorVar() == access.getTensorVar() && argAccess.getIndexVars() == access.getIndexVars()) {
      foundAccess = true;
      break;
    }
  }
  if (!foundAccess) {
    taco_uerror << "Access: " << access << " does not appear in index statement as an argument";
  }

  // check access is correct
  ProvenanceGraph provGraph = ProvenanceGraph(*this);
  vector<IndexVar> underivedParentAncestors = provGraph.getUnderivedAncestors(i);
  size_t max_mode = 0;
  for (IndexVar underived : underivedParentAncestors) {
    size_t mode_index = 0; // which of the access index vars match?
    for (auto var : access.getIndexVars()) {
      if (var == underived) {
        break;
      }
      mode_index++;
    }
    if (mode_index > max_mode) max_mode = mode_index;
  }
  if ((size_t)max_mode >= access.getIndexVars().size()) {
    taco_uerror << "Index variable " << i << " does not appear in access: " << access;
  }

  int mode = access.getTensorVar().getFormat().getModeOrdering()[max_mode];
  if (access.getTensorVar().getFormat().getModeFormats()[mode] == Dense) {
    taco_uerror << "Pos transformation is not valid for dense formats, the coordinate space should be transformed instead";
  }

  IndexVarRel rel = IndexVarRel(new PosRelNode(i, ipos, access));
  string reason;

  // Add predicate to concrete index notation
  IndexStmt transformed = Transformation(AddSuchThatPredicates({rel})).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  // Replace all occurrences of i with ipos
  transformed = Transformation(ForAllReplace({i}, {ipos})).apply(transformed, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  return transformed;
}

IndexStmt IndexStmt::fuse(IndexVar i, IndexVar j, IndexVar f) const {
  IndexVarRel rel = IndexVarRel(new FuseRelNode(i, j, f));
  string reason;

  // Add predicate to concrete index notation
  IndexStmt transformed = Transformation(AddSuchThatPredicates({rel})).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  // Replace all occurrences of i, j with f
  transformed = Transformation(ForAllReplace({i,j}, {f})).apply(transformed, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  return transformed;
}

IndexStmt IndexStmt::bound(IndexVar i, IndexVar i1, size_t bound, BoundType bound_type) const {
  IndexVarRel rel = IndexVarRel(new BoundRelNode(i, i1, bound, bound_type));
  string reason;

  // Add predicate to concrete index notation
  IndexStmt transformed = Transformation(AddSuchThatPredicates({rel})).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  // Replace all occurrences of i with i1
  transformed = Transformation(ForAllReplace({i}, {i1})).apply(transformed, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  return transformed;
}

IndexStmt IndexStmt::unroll(IndexVar i, size_t unrollFactor) const {
  struct UnrollLoop : IndexNotationRewriter {
    using IndexNotationRewriter::visit;
    IndexVar i;
    size_t unrollFactor;
    UnrollLoop(IndexVar i, size_t unrollFactor) : i(i), unrollFactor(unrollFactor) {}

    void visit(const ForallNode* node) {
      if (node->indexVar == i) {
        stmt = Forall(i, rewrite(node->stmt), node->merge_strategy, node->parallel_unit, node->output_race_strategy, unrollFactor);
      }
      else {
        IndexNotationRewriter::visit(node);
      }
    }
  };
  return UnrollLoop(i, unrollFactor).rewrite(*this);
}

IndexStmt IndexStmt::assemble(TensorVar result, AssembleStrategy strategy,
                              bool separatelySchedulable) const {
  string reason;
  IndexStmt transformed = 
      SetAssembleStrategy(result, strategy, 
                          separatelySchedulable).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }
  return transformed;
}

std::ostream& operator<<(std::ostream& os, const IndexStmt& expr) {
  if (!expr.defined()) return os << "IndexStmt()";
  IndexNotationPrinter printer(os);
  printer.print(expr);
  return os;
}

// class Assignment
Assignment::Assignment(const AssignmentNode* n) : IndexStmt(n) {
}

Assignment::Assignment(Access lhs, IndexExpr rhs, IndexExpr op)
    : Assignment(new AssignmentNode(lhs, rhs, op)) {
}

Assignment::Assignment(TensorVar tensor, vector<IndexVar> indices,
                       IndexExpr rhs, IndexExpr op,
                       const std::map<int, std::shared_ptr<IndexVarIterationModifier>>& modifiers)
    : Assignment(Access(tensor, indices, modifiers), rhs, op) {
}

Access Assignment::getLhs() const {
  return getNode(*this)->lhs;
}

IndexExpr Assignment::getRhs() const {
  return getNode(*this)->rhs;
}

IndexExpr Assignment::getOperator() const {
  return getNode(*this)->op;
}

const std::vector<IndexVar>& Assignment::getFreeVars() const {
  return getLhs().getIndexVars();
}

std::vector<IndexVar> Assignment::getReductionVars() const {
  vector<IndexVar> freeVars = getLhs().getIndexVars();
  set<IndexVar> seen(freeVars.begin(), freeVars.end());
  vector<IndexVar> reductionVars;
  match(getRhs(),
    std::function<void(const AccessNode*)>([&](const AccessNode* op) {
    for (auto& var : op->indexVars) {
      if (!util::contains(seen, var)) {
        reductionVars.push_back(var);
        seen.insert(var);
      }
    }
    })
  );
  return reductionVars;
}

template <> bool isa<Assignment>(IndexStmt s) {
  return isa<AssignmentNode>(s.ptr);
}

template <> Assignment to<Assignment>(IndexStmt s) {
  taco_iassert(isa<Assignment>(s));
  return Assignment(to<AssignmentNode>(s.ptr));
}


// class Yield
Yield::Yield(const YieldNode* n) : IndexStmt(n) {
}

Yield::Yield(const std::vector<IndexVar>& indexVars, IndexExpr expr)
    : Yield(new YieldNode(indexVars, expr)) {
}

const std::vector<IndexVar>& Yield::getIndexVars() const {
  return getNode(*this)->indexVars;
}

IndexExpr Yield::getExpr() const {
  return getNode(*this)->expr;
}


// class Forall
Forall::Forall(const ForallNode* n) : IndexStmt(n) {
}

Forall::Forall(IndexVar indexVar, IndexStmt stmt)
    : Forall(indexVar, stmt, MergeStrategy::TwoFinger, ParallelUnit::NotParallel, OutputRaceStrategy::IgnoreRaces) {
}

Forall::Forall(IndexVar indexVar, IndexStmt stmt, MergeStrategy merge_strategy, ParallelUnit parallel_unit, OutputRaceStrategy output_race_strategy, size_t unrollFactor)
        : Forall(new ForallNode(indexVar, stmt, merge_strategy, parallel_unit, output_race_strategy, unrollFactor)) {
}

IndexVar Forall::getIndexVar() const {
  return getNode(*this)->indexVar;
}

IndexStmt Forall::getStmt() const {
  return getNode(*this)->stmt;
}

ParallelUnit Forall::getParallelUnit() const {
  return getNode(*this)->parallel_unit;
}

OutputRaceStrategy Forall::getOutputRaceStrategy() const {
  return getNode(*this)->output_race_strategy;
}

MergeStrategy Forall::getMergeStrategy() const {
  return getNode(*this)->merge_strategy;
}

size_t Forall::getUnrollFactor() const {
  return getNode(*this)->unrollFactor;
}

Forall forall(IndexVar i, IndexStmt stmt) {
  return Forall(i, stmt);
}

Forall forall(IndexVar i, IndexStmt stmt, MergeStrategy merge_strategy, ParallelUnit parallel_unit, OutputRaceStrategy output_race_strategy, size_t unrollFactor) {
  return Forall(i, stmt, merge_strategy, parallel_unit, output_race_strategy, unrollFactor);
}

template <> bool isa<Forall>(IndexStmt s) {
  return isa<ForallNode>(s.ptr);
}

template <> Forall to<Forall>(IndexStmt s) {
  taco_iassert(isa<Forall>(s));
  return Forall(to<ForallNode>(s.ptr));
}

//class ForallMany
ForallMany::ForallMany(const ForallManyNode* n) : IndexStmt(n){
}

ForallMany::ForallMany(IndexVar indexVar, std::vector<IndexStmt> stmts)
  : ForallMany(new ForallManyNode(indexVar, stmts)) {
}

ForallMany::ForallMany(std::vector<IndexStmt> stmts) : ForallMany(IndexVar(), stmts){  
}

IndexVar ForallMany::getIndexVar() const{
  return getNode(*this)->indexVar;

}
std::vector<IndexStmt> ForallMany::getStmts() const{
  return getNode(*this)->stmts;
}

template <> bool isa<ForallMany>(IndexStmt s) {
  return isa<ForallMany>(s.ptr);
}

template <> ForallMany to<ForallMany>(IndexStmt s) {
  taco_iassert(isa<ForallMany>(s));
  return ForallMany(to<ForallManyNode>(s.ptr));
}

// class Where
Where::Where(const WhereNode* n) : IndexStmt(n) {
}

Where::Where(IndexStmt consumer, IndexStmt producer)
    : Where(new WhereNode(consumer, producer)) {
}

IndexStmt Where::getConsumer() {
  return getNode(*this)->consumer;
}


IndexStmt Where::getProducer() {
  return getNode(*this)->producer;
}

TensorVar Where::getResult() {
  return getResultAccesses(getConsumer()).first[0].getTensorVar();
}

TensorVar Where::getTemporary() {
  return getResultAccesses(getProducer()).first[0].getTensorVar();
}

Where where(IndexStmt consumer, IndexStmt producer) {
  return Where(consumer, producer);
}

template <> bool isa<Where>(IndexStmt s) {
  return isa<WhereNode>(s.ptr);
}

template <> Where to<Where>(IndexStmt s) {
  taco_iassert(isa<Where>(s));
  return Where(to<WhereNode>(s.ptr));
}

// class accelerate
Accelerate::Accelerate(const AccelerateNode* n) : IndexStmt(n) {
}

Accelerate::Accelerate(IndexStmt consumer, IndexStmt producer, ConcreteAccelerateCodeGenerator accelGen)
    : Accelerate(new AccelerateNode(consumer, producer, accelGen)) {
}

IndexStmt Accelerate::getConsumer() {
  return getNode(*this)->consumer;
}


IndexStmt Accelerate::getProducer() {
  return getNode(*this)->producer;
}

ConcreteAccelerateCodeGenerator Accelerate::getAccelGen() {
  return getNode(*this)->accelGen;
}

TensorVar Accelerate::getResult() {
  return getResultAccesses(getConsumer()).first[0].getTensorVar();
}

TensorVar Accelerate::getTemporary() {
  return getResultAccesses(getProducer()).first[0].getTensorVar();
}

Accelerate accelerate(IndexStmt consumer, IndexStmt producer, ConcreteAccelerateCodeGenerator accelGen) {
  return Accelerate(consumer, producer, accelGen);
}

template <> bool isa<Accelerate>(IndexStmt s) {
  return isa<AccelerateNode>(s.ptr);
}

template <> Accelerate to<Accelerate>(IndexStmt s) {
  taco_iassert(isa<Accelerate>(s));
  return Accelerate(to<AccelerateNode>(s.ptr));
}

DimReduction::DimReduction(const DimReductionNode* n) : IndexStmt(n){
}

DimReduction::DimReduction(IndexStmt consumer, IndexStmt producer, std::vector<TensorVar> temps)
    : DimReduction(new DimReductionNode(consumer, producer, temps)){
}

IndexStmt DimReduction::getConsumer(){
  return getNode(*this)->consumer;
}

IndexStmt DimReduction::getProducer(){
  return getNode(*this)->producer;
}

std::vector<TensorVar> DimReduction::getTemporaries(){
  return getNode(*this)->temps;
}

template <> bool isa<DimReduction>(IndexStmt s) {
  return isa<DimReductionNode>(s.ptr);
}

template <> DimReduction to<DimReduction>(IndexStmt s) {
  taco_iassert(isa<DimReduction>(s));
  return DimReduction(to<DimReductionNode>(s.ptr));
}

//class InterfaceCall
InterfaceCall::InterfaceCall(const InterfaceCallNode* n) : IndexStmt(n) {
}

InterfaceCall::InterfaceCall(Assignment producer, ConcreteAccelerateCodeGenerator accelGen, TensorVar temp)
  : InterfaceCall(new InterfaceCallNode(producer, accelGen, temp)){
}

Assignment InterfaceCall::getProducer(){
  return getNode(*this)->producer;
}

ConcreteAccelerateCodeGenerator InterfaceCall::getAccelGen(){
  return getNode(*this)->codeGen;
}

TensorVar InterfaceCall::getTemporary(){
  return getNode(*this)->temp;
}

template <> bool isa<InterfaceCall>(IndexStmt s) {
  return isa<InterfaceCallNode>(s.ptr);
}

template <> InterfaceCall to<InterfaceCall>(IndexStmt s) {
  taco_iassert(isa<InterfaceCall>(s));
  return InterfaceCall(to<InterfaceCallNode>(s.ptr));
}

// class Sequence
Sequence::Sequence(const SequenceNode* n) :IndexStmt(n) {
}

Sequence::Sequence(IndexStmt definition, IndexStmt mutation)
    : Sequence(new SequenceNode(definition, mutation)) {
}

IndexStmt Sequence::getDefinition() const {
  return getNode(*this)->definition;
}

IndexStmt Sequence::getMutation() const {
  return getNode(*this)->mutation;
}

Sequence sequence(IndexStmt definition, IndexStmt mutation) {
  return Sequence(definition, mutation);
}

template <> bool isa<Sequence>(IndexStmt s) {
  return isa<SequenceNode>(s.ptr);
}

template <> Sequence to<Sequence>(IndexStmt s) {
  taco_iassert(isa<Sequence>(s));
  return Sequence(to<SequenceNode>(s.ptr));
}


// class Assemble
Assemble::Assemble(const AssembleNode* n) :IndexStmt(n) {
}

Assemble::Assemble(IndexStmt queries, IndexStmt compute, 
                   AttrQueryResults results)
    : Assemble(new AssembleNode(queries, compute, results)) {
}

IndexStmt Assemble::getQueries() const {
  return getNode(*this)->queries;
}

IndexStmt Assemble::getCompute() const {
  return getNode(*this)->compute;
}

const Assemble::AttrQueryResults& Assemble::getAttrQueryResults() const {
  return getNode(*this)->results;
}

Assemble assemble(IndexStmt queries, IndexStmt compute, 
                  Assemble::AttrQueryResults results) {
  return Assemble(queries, compute, results);
}

template <> bool isa<Assemble>(IndexStmt s) {
  return isa<AssembleNode>(s.ptr);
}

template <> Assemble to<Assemble>(IndexStmt s) {
  taco_iassert(isa<Assemble>(s));
  return Assemble(to<AssembleNode>(s.ptr));
}


// class Multi
Multi::Multi(const MultiNode* n) : IndexStmt(n) {
}

Multi::Multi(IndexStmt stmt1, IndexStmt stmt2)
    : Multi(new MultiNode(stmt1, stmt2)) {
}

IndexStmt Multi::getStmt1() const {
  return getNode(*this)->stmt1;
}

IndexStmt Multi::getStmt2() const {
  return getNode(*this)->stmt2;
}

Multi multi(IndexStmt stmt1, IndexStmt stmt2) {
  return Multi(stmt1, stmt2);
}

template <> bool isa<Multi>(IndexStmt s) {
  return isa<MultiNode>(s.ptr);
}

template <> Multi to<Multi>(IndexStmt s) {
  taco_iassert(isa<Multi>(s));
  return Multi(to<MultiNode>(s.ptr));
}

// class SuchThat
SuchThat::SuchThat(const SuchThatNode* n) : IndexStmt(n) {
}

SuchThat::SuchThat(IndexStmt stmt, std::vector<IndexVarRel> predicate)
        : SuchThat(new SuchThatNode(stmt, predicate)) {
}

IndexStmt SuchThat::getStmt() const {
  return getNode(*this)->stmt;
}

std::vector<IndexVarRel> SuchThat::getPredicate() const {
  return getNode(*this)->predicate;
}

SuchThat suchthat(IndexStmt stmt, std::vector<IndexVarRel> predicate) {
  return SuchThat(stmt, predicate);
}

template <> bool isa<SuchThat>(IndexStmt s) {
  return isa<SuchThatNode>(s.ptr);
}

template <> SuchThat to<SuchThat>(IndexStmt s) {
  taco_iassert(isa<SuchThat>(s));
  return SuchThat(to<SuchThatNode>(s.ptr));
}

std::ostream& operator<<(std::ostream& os, const IndexObject& op){

  return op.getNode()->print(os);

}

//class dynamic order
struct DynamicOrder::Content {
  int min;
  int max;
  std::string name;
};

DynamicOrder::DynamicOrder() : DynamicOrder(util::uniqueName('I')){
}

DynamicOrder::DynamicOrder(std::string name) : content(new Content) {
  content->min = -1;
  content->max = -1;
  content->name = name;
}

void DynamicOrder::setMin(int min){
  taco_uassert(min > 0);
  content->min = min;
}

void DynamicOrder::setMax(int max){
  taco_uassert(max > 0);
  content->max = max;
}

bool DynamicOrder::hasMin() const{
  return (content->min != -1);
}

bool DynamicOrder::hasMax() const{
  return (content->max != -1);
}

bool DynamicOrder::hasFixedSize() const{
  if (!hasMin() || !hasMax()){
    return false;
  }
  return getMin() == getMax();
}

int DynamicOrder::getMin() const{
  return content->min;
}

int DynamicOrder::getMax() const{
  return content->max;
}

void DynamicOrder::setSize(int size){
   content->max = size;
   content->min = size;
}

std::string DynamicOrder::getName() const{
  return content->name;
}

std::ostream& operator<<(std::ostream& os, const DynamicOrder& op){

  if (op.hasMin()){
    os << "min(" << op.getMin() << ") ";
  }
  os << "...";
  if (op.hasMax()){
    os << " max(" << op.getMax() << ")";
  }
  return os;
}

bool operator<(const DynamicOrder& a, const DynamicOrder& b) {
  return a.content < b.content;
}

bool operator==(const DynamicOrder& a, const DynamicOrder& b) {
  return a.content < b.content;
}

std::ostream& DynamicOrder::print(std::ostream& os) const {
  return os << *this;
}

DynamicIndexAccess DynamicOrder::operator()(const DynamicIndexIterator& index){
  if (getName() != index.getDynamicOrder().getName()){
    taco_uerror << "Indexing into a dynmaic order that the iterator was not initialized with";
  }
  return DynamicIndexAccess(index);
}

DynamicIndexAccess DynamicOrder::operator()(const DynamicIndexIterator& index) const{
  if (getName() != index.getDynamicOrder().getName()){
    taco_uerror << "Indexing into a dynmaic order that the iterator was not initialized with";
  }
  return DynamicIndexAccess(index);
}

// class IndexVar
IndexVar::IndexVar() : IndexVar(util::uniqueName('i')) {}

IndexVar::IndexVar(const std::string& name) : IndexVar(name, Datatype::Int32) {}

IndexVar::IndexVar(const std::string& name, const Datatype& type) : IndexVar(new IndexVarNode(name, type)) {}

IndexVar::IndexVar(const IndexVarNode* n) : IndexExpr(n) {}

template <> bool isa<IndexVar>(IndexExpr e) {
  return isa<IndexVarNode>(e.ptr);
}

template <> IndexVar to<IndexVar>(IndexExpr e) {
  taco_iassert(isa<IndexVar>(e));
  return IndexVar(to<IndexVarNode>(e.ptr));
}

std::string IndexVar::getName() const {
  return getNode(*this)->getName();
}

std::ostream& IndexVar::print(std::ostream& os) const {
  return os << *this;
}

WindowedIndexVar IndexVar::operator()(int lo, int hi, int stride) {
  return WindowedIndexVar(*this, lo, hi, stride);
}

IndexSetVar IndexVar::operator()(std::vector<int>&& indexSet) {
  return IndexSetVar(*this, indexSet);
}

IndexSetVar IndexVar::operator()(std::vector<int>& indexSet) {
  return IndexSetVar(*this, indexSet);
}

bool operator==(const IndexVar& a, const IndexVar& b) {
  return *getNode(a) == *getNode(b);
}

bool operator<(const IndexVar& a, const IndexVar& b) {
  return *getNode(a) < *getNode(b);
}

bool operator!=(const IndexVar& a , const IndexVar& b) {
  return *getNode(a) != *getNode(b);
}

bool operator>=(const IndexVar& a, const IndexVar& b) {
  return *getNode(a) >= *getNode(b);
}

bool operator<=(const IndexVar& a, const IndexVar& b) {
  return *getNode(a) <= *getNode(b);
}

bool operator>(const IndexVar& a , const IndexVar& b) {
  return *getNode(a) > *getNode(b);
}

std::ostream& operator<<(std::ostream& os, const std::shared_ptr<IndexVarInterface>& var) {
  std::stringstream ss;
  IndexVarInterface::match(var, [&](std::shared_ptr<IndexVar> ivar) {
    ss << *ivar;
  }, [&](std::shared_ptr<WindowedIndexVar> wvar) {
    ss << *wvar;
  }, [&](std::shared_ptr<IndexSetVar> svar) {
    ss << *svar;
  });
  return os << ss.str();
}

std::ostream& operator<<(std::ostream& os, const IndexVar& var) {
  return os << var.getName();
}

std::ostream& operator<<(std::ostream& os, const WindowedIndexVar& var) {
  return os << var.getIndexVar();
}

std::ostream& operator<<(std::ostream& os, const IndexSetVar& var) {
  return os << var.getIndexVar();
}

WindowedIndexVar::WindowedIndexVar(IndexVar base, int lo, int hi, int stride) : content( new Content){
  this->content->base = base;
  this->content->lo = lo;
  this->content->hi = hi;
  this->content->stride = stride;
}

IndexVar WindowedIndexVar::getIndexVar() const {
  return this->content->base;
}

int WindowedIndexVar::getLowerBound() const {
  return this->content->lo;
}

int WindowedIndexVar::getUpperBound() const {
  return this->content->hi;
}

int WindowedIndexVar::getStride() const {
  return this->content->stride;
}

int WindowedIndexVar::getWindowSize() const {
  return (this->content->hi - this->content->lo) / this->content->stride;
}

IndexSetVar::IndexSetVar(IndexVar base, std::vector<int> indexSet): content (new Content) {
  this->content->base = base;
  this->content->indexSet = indexSet;
}

IndexVar IndexSetVar::getIndexVar() const {
  return this->content->base;
}

const std::vector<int>& IndexSetVar::getIndexSet() const {
  return this->content->indexSet;
}

// class TensorVar
struct TensorVar::Content {
  int id;
  string name;
  Type type;
  Format format;
  Schedule schedule;
  Literal fill;
  std::set<std::string> properties;
};

TensorVar::TensorVar() : content(nullptr) {
}

static Format createDenseFormat(const Type& type) {
  return Format(vector<ModeFormatPack>(type.getOrder(), ModeFormat(Dense)));
}

TensorVar::TensorVar(const Type& type, const Literal& fill)
: TensorVar(type, createDenseFormat(type), fill) {
}

TensorVar::TensorVar(const std::string& name, const Type& type, const Literal& fill)
: TensorVar(-1, name, type, createDenseFormat(type), fill) {
}

TensorVar::TensorVar(const Type& type, const Format& format, const Literal& fill)
    : TensorVar(-1, util::uniqueName('A'), type, format, fill) {
}

TensorVar::TensorVar(const string& name, const Type& type, const Format& format, const Literal& fill)
    : TensorVar(-1, name, type, format, fill) {
}

TensorVar::TensorVar(const int& id, const string& name, const Type& type, const Format& format, const Literal& fill)
    : content(new Content) {
  content->id = id;
  content->name = name;
  content->type = type;
  content->format = format;
  content->fill = fill.defined()? fill : Literal::zero(type.getDataType());
}

int TensorVar::getId() const {
  return content->id;
}

std::string TensorVar::getName() const {
  return content->name;
}

int TensorVar::getOrder() const {
  return content->type.getShape().getOrder();
}

Shape TensorVar::getShape() const {
  return content->type.getShape();
}

const Type& TensorVar::getType() const {
  return content->type;
}

const Format& TensorVar::getFormat() const {
  return content->format;
}

const Schedule& TensorVar::getSchedule() const {
  struct GetSchedule : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;
    Schedule schedule;
    void visit(const BinaryExprNode* expr) {
      auto workspace = expr->getWorkspace();
      if (workspace.defined()) {
        schedule.addPrecompute(workspace);
      }
    }
  };
  GetSchedule getSchedule;
  content->schedule.clearPrecomputes();
  getSchedule.schedule = content->schedule;
  return content->schedule;
}

const Literal& TensorVar::getFill() const {
  return content->fill;
}

void TensorVar::setFill(const Literal &fill) {
  content->fill = fill;
}

void TensorVar::setName(std::string name) {
  content->name = name;
}

bool TensorVar::defined() const {
  return content != nullptr;
}

void TensorVar::setProperty(std::string property) const{
  content->properties.insert(property);
}

void TensorVar::eraseAllProperties() const{
  content->properties.clear();
}

void TensorVar::eraseProperty(std::string property) const{
   content->properties.erase(property);
}

bool TensorVar::hasProperty(std::string property) const{
  return content->properties.count(property);
}

bool TensorVar::hasProperties(std::set<std::string> desiredProperties) const{
  return std::includes(content->properties.begin(), content->properties.end(), 
                       desiredProperties.begin(), desiredProperties.end());
}

std::set<std::string> TensorVar::getProperties() const{
  return content->properties;
}

const Access TensorVar::operator()(const std::vector<IndexVar>& indices) const {
  taco_uassert((int)indices.size() == getOrder()) <<
      "A tensor of order " << getOrder() << " must be indexed with " <<
      getOrder() << " variables, but is indexed with:  " << util::join(indices);
  return Access(new AccessNode(*this, indices, {}, false));
}

Access TensorVar::operator()(const std::vector<IndexVar>& indices) {
  taco_uassert((int)indices.size() == getOrder()) <<
      "A tensor of order " << getOrder() << " must be indexed with " <<
      getOrder() << " variables, but is indexed with:  " << util::join(indices);
  return Access(new AccessNode(*this, indices, {}, false));
}

Assignment TensorVar::operator=(IndexExpr expr) {
  taco_uassert(getOrder() == 0)
      << "Must use index variable on the left-hand-side when assigning an "
      << "expression to a non-scalar tensor.";
  Assignment assignment = Assignment(*this, {}, expr);
  check(assignment);
  return assignment;
}

Assignment TensorVar::operator+=(IndexExpr expr) {
  taco_uassert(getOrder() == 0)
      << "Must use index variable on the left-hand-side when assigning an "
      << "expression to a non-scalar tensor.";
  Assignment assignment = Assignment(*this, {}, expr, new AddNode);
  check(assignment);
  return assignment;
}

bool operator==(const TensorVar& a, const TensorVar& b) {
  return a.content == b.content;
}

bool operator<(const TensorVar& a, const TensorVar& b) {
  return a.content < b.content;
}

std::ostream& operator<<(std::ostream& os, const TensorVar& var) {
  return os << var.getName() << " : " << var.getType();
}


static bool isValid(Assignment assignment, string* reason) {
  if (reason == nullptr) {
    INIT_REASON(reason);
  }
  auto rhs = assignment.getRhs();
  auto lhs = assignment.getLhs();
  auto result = lhs.getTensorVar();
  auto freeVars = lhs.getIndexVars();
  auto shape = result.getType().getShape();

  // If the LHS access has any windowed modes, use the dimensions of those
  // windows as the shape, rather than the shape of the underlying tensor.
  if (lhs.hasWindowedModes() || lhs.hasIndexSetModes()) {
    vector<Dimension> dims(shape.getOrder());
    for (int i = 0; i < shape.getOrder();i++) {
      dims[i] = shape.getDimension(i);
      if (lhs.isModeWindowed(i)) {
        dims[i] = Dimension(lhs.getWindowSize(i));
      } else if (lhs.isModeIndexSet(i)) {
        dims[i] = Dimension(lhs.getIndexSet(i).size());
      }
    }
    shape = Shape(dims);
  }

  auto typecheck = error::dimensionsTypecheck(freeVars, rhs, shape);
  if (!typecheck.first) {
    *reason = error::expr_dimension_mismatch + " " + typecheck.second;
    return false;
  }
  return true;
}

// functions
bool isEinsumNotation(IndexStmt stmt, std::string* reason) {
  INIT_REASON(reason);

  if (!isa<Assignment>(stmt)) {
    *reason = "Einsum notation statements must be assignments.";
    return false;
  }

  if (!isValid(to<Assignment>(stmt), reason)) {
    return false;
  }

  // Einsum notation until proved otherwise
  bool isEinsum = true;

  // Additions are not allowed under the first multiplication
  bool mulnodeVisited = false;

  match(stmt,
    std::function<void(const AddNode*,Matcher*)>([&](const AddNode* op,
                                                     Matcher* ctx) {
      if (mulnodeVisited) {
        *reason = "additions in einsum notation must not be nested under "
                  "multiplications";
        isEinsum = false;
      }
      else {
        ctx->match(op->a);
        ctx->match(op->b);
      }
    }),
    std::function<void(const SubNode*,Matcher*)>([&](const SubNode* op,
                                                     Matcher* ctx) {
      if (mulnodeVisited) {
        *reason = "subtractions in einsum notation must not be nested under "
                  "multiplications";
        isEinsum = false;
      }
      else {
        ctx->match(op->a);
        ctx->match(op->b);
      }
    }),
    std::function<void(const MulNode*,Matcher*)>([&](const MulNode* op,
                                                     Matcher* ctx) {
      bool topMulNode = !mulnodeVisited;
      mulnodeVisited = true;
      ctx->match(op->a);
      ctx->match(op->b);
      if (topMulNode) {
        mulnodeVisited = false;
      }
    }),
    std::function<void(const BinaryExprNode*)>([&](const BinaryExprNode* op) {
      *reason = "einsum notation may not contain " + op->getOperatorString() +
                " operations";
      isEinsum = false;
    }),
    std::function<void(const ReductionNode*)>([&](const ReductionNode* op) {
      *reason = "einsum notation may not contain reductions";
      isEinsum = false;
    })
  );
  return isEinsum;
}

bool isReductionNotation(IndexStmt stmt, std::string* reason) {
  INIT_REASON(reason);

  if (!isa<Assignment>(stmt)) {
    *reason = "reduction notation statements must be assignments";
    return false;
  }

  if (!isValid(to<Assignment>(stmt), reason)) {
    return false;
  }

  // Reduction notation until proved otherwise
  bool isReduction = true;

  util::ScopedSet<IndexVar> boundVars; 
  vector<IndexVar> boundVarsList;
  for (auto& var : to<Assignment>(stmt).getFreeVars()) {
    boundVars.insert({var});
    boundVarsList.push_back(var);
  }

  match(stmt,
    std::function<void(const ReductionNode*,Matcher*)>([&](
        const ReductionNode* op, Matcher* ctx) {
      boundVars.scope();
      boundVars.insert({op->var});
      ctx->match(op->a);
      boundVars.unscope();
    }),
    std::function<void(const AccessNode*)>([&](const AccessNode* op) {
      for (auto& var : op->indexVars) {
        if (!boundVars.contains(var)) {
          *reason = "all reduction variables in reduction notation must be "
                    "bound by a reduction expression";
          isReduction = false;
        }
      }
    })
  );

  return isReduction;
}

bool isReductionNotationScheduled(IndexStmt stmt, ProvenanceGraph provGraph, std::string* reason) {
  INIT_REASON(reason);

  if (!isa<Assignment>(stmt)) {
    *reason = "reduction notation statements must be assignments";
    return false;
  }

  if (!isValid(to<Assignment>(stmt), reason)) {
    return false;
  }

  // Reduction notation until proved otherwise
  bool isReduction = true;

  util::ScopedSet<IndexVar> boundVars; 
  vector<IndexVar> boundVarsList;
  for (auto& var : to<Assignment>(stmt).getFreeVars()) {
    boundVars.insert({var});
    boundVarsList.push_back(var);
  }

  match(stmt,
        std::function<void(const ReductionNode*,Matcher*)>([&](
          const ReductionNode* op, Matcher* ctx) {
          boundVars.scope();
          boundVars.insert({op->var});
          ctx->match(op->a);
          boundVars.unscope();
        }),
        std::function<void(const AccessNode*)>([&](const AccessNode* op) {
          for (auto& var : op->indexVars) {
            if (!boundVars.contains(var)) {
              // This detects to see if one of the boundVars is an ancestor of var
              // or if boundVars is a descendant of var given the Provenance Graph.
              // If either of these are true, then the statement is still in reduction notation.
              if (provGraph.isFullyDerived(var)) {
                auto ancestors = provGraph.getUnderivedAncestors(var);
                for (auto& ancestor: ancestors) {
                  if (boundVars.contains(ancestor)) {
                    isReduction = true;
                  }
                }
              } else {
                auto descendants = provGraph.getFullyDerivedDescendants(var);
                for (auto& descendant : descendants) {
                  if (boundVars.contains(descendant)) {
                    isReduction = true;
                  }
                }
              }
                  *reason = "all reduction variables in reduction notation must be "
                            "bound by a reduction expression";
              isReduction = false;
            }
          }
        })
  );
  return isReduction;
}

bool isConcreteNotation(IndexStmt stmt, std::string* reason) {
  taco_iassert(stmt.defined()) << "the index statement is undefined";
  INIT_REASON(reason);

  // Concrete notation until proved otherwise
  bool isConcrete = true;

  bool inWhereProducer = false;
  bool inWhereConsumer = false;
  util::ScopedSet<IndexVar> boundVars; 
  std::set<IndexVar> definedVars; // used to check if all variables recoverable TODO: need to actually use scope like above

  ProvenanceGraph provGraph = ProvenanceGraph(stmt);

  match(stmt,
    std::function<void(const ForallNode*,Matcher*)>([&](const ForallNode* op,
                                                        Matcher* ctx) {
      boundVars.scope();
      boundVars.insert({op->indexVar});
      definedVars.insert(op->indexVar);
      ctx->match(op->stmt);
      boundVars.unscope();
    }),
    std::function<void(const AccessNode*)>([&](const AccessNode* op) {
      for (auto& var : op->indexVars) {
        // non underived variables may appear in temporaries, but we don't check these
        if (!boundVars.contains(var) && provGraph.isUnderived(var) &&
           (provGraph.isFullyDerived(var) || !provGraph.isRecoverable(var, definedVars))) {
          *reason = "all variables in concrete notation must be bound by a "
                    "forall statement";
          isConcrete = false;
        }
      }
    }),
    std::function<void(const IndexVarNode*)>([&](const IndexVarNode* op) {
      IndexVar var(op);
      if (!boundVars.contains(var) && provGraph.isUnderived(var) &&
         (provGraph.isFullyDerived(var) || !provGraph.isRecoverable(var, definedVars)))  {
        *reason = "index variables used in compute statements must be nested under a forall";
        isConcrete = false;
      }
    }),
    std::function<void(const WhereNode*,Matcher*)>([&](const WhereNode* op, Matcher* ctx) {
      bool alreadyInProducer = inWhereProducer;
      inWhereProducer = true;
      ctx->match(op->producer);
      if (!alreadyInProducer) inWhereProducer = false;
      bool alreadyInConsumer = inWhereConsumer;
      inWhereConsumer = true;
      ctx->match(op->consumer);
      if (!alreadyInConsumer) inWhereConsumer = false;
    }),
    std::function<void(const AssignmentNode*,Matcher*)>([&](
        const AssignmentNode* op, Matcher* ctx) {
      if(!inWhereConsumer && !inWhereProducer && !isValid(Assignment(op), reason)) { // TODO: fix check for precompute
        isConcrete = false;
        return;
      }

      // Handles derived vars on RHS with underived vars on LHS.
      Assignment assignPtrWrapper = Assignment(op);
      std::vector<IndexVar> possibleReductionVars = assignPtrWrapper.getReductionVars();
      std::vector<IndexVar> freeVars = assignPtrWrapper.getFreeVars();
      std::set<IndexVar> freeVarsSet(freeVars.begin(), freeVars.end());

      int numReductionVars = 0;
      for(const auto& reductionVar : possibleReductionVars) {
        std::vector<IndexVar> underivedParents = provGraph.getUnderivedAncestors(reductionVar);
        for(const auto& parent : underivedParents) {
          if(!util::contains(freeVarsSet, parent)) {
            ++numReductionVars;
          }
        }
      }
      // allow introducing precompute loops where we set a temporary to values instead of +=
      if (numReductionVars > 0 &&
          op->op == IndexExpr() && !inWhereProducer) {
        *reason = "reduction variables in concrete notation must be dominated "
                  "by compound assignments (such as +=)";
        isConcrete = false;
        return;
      }
      ctx->match(op->lhs);
      ctx->match(op->rhs);
    }),
    std::function<void(const ReductionNode*)>([&](const ReductionNode* op) {
      *reason = "concrete notation cannot contain reduction nodes";
      isConcrete = false;
    }),
    std::function<void(const SuchThatNode*)>([&](const SuchThatNode* op) {
      const string failed_reason = "concrete notation cannot contain nested SuchThat nodes";
      if (!isa<SuchThat>(stmt)) {
        *reason = failed_reason;
        isConcrete = false;
        return;
      }
      SuchThat firstSuchThat = to<SuchThat>(stmt);
      if (firstSuchThat != op) {
        *reason = failed_reason;
        isConcrete = false;
        return;
      }
    })
  );
  return isConcrete;
}

Assignment makeReductionNotation(Assignment assignment) {
  IndexExpr expr = assignment.getRhs();
  std::vector<IndexVar> free = assignment.getLhs().getIndexVars();
  if (!isEinsumNotation(assignment)) {
    return assignment;
  }

  struct MakeReductionNotation : IndexNotationRewriter {
    MakeReductionNotation(const std::vector<IndexVar>& free)
        : free(free.begin(), free.end()){}

    std::set<IndexVar> free;
    bool onlyOneTerm;

    IndexExpr addReductions(IndexExpr expr) {
      auto vars = getIndexVars(expr);
      for (auto& var : util::reverse(vars)) {
        if (!util::contains(free, var)) {
          expr = sum(var,expr);
        }
      }
      return expr;
    }

    IndexExpr einsum(const IndexExpr& expr) {
      onlyOneTerm = true;
      IndexExpr einsumexpr = rewrite(expr);

      if (onlyOneTerm) {
        einsumexpr = addReductions(einsumexpr);
      }

      return einsumexpr;
    }

    using IndexNotationRewriter::visit;

    void visit(const AddNode* op) {
      // Sum every reduction variables over each term
      onlyOneTerm = false;

      IndexExpr a = addReductions(op->a);
      IndexExpr b = addReductions(op->b);
      if (a == op->a && b == op->b) {
        expr = op;
      }
      else {
        expr = new AddNode(a, b);
      }
    }

    void visit(const SubNode* op) {
      // Sum every reduction variables over each term
      onlyOneTerm = false;

      IndexExpr a = addReductions(op->a);
      IndexExpr b = addReductions(op->b);
      if (a == op->a && b == op->b) {
        expr = op;
      }
      else {
        expr = new SubNode(a, b);
      }
    }
  };
  return Assignment(assignment.getLhs(),
                    MakeReductionNotation(free).einsum(expr),
                    assignment.getOperator());
}

IndexStmt makeReductionNotation(IndexStmt stmtOriginal) {
  taco_iassert(isEinsumNotation(stmtOriginal));
  if (!isa<Assignment>(stmtOriginal)){
    struct MakeReductionNotation : IndexNotationRewriter {
      MakeReductionNotation() = default;

      IndexStmt rewriteReduction(const IndexStmt& s) {
        IndexStmt rewriteReductionStmt = rewrite(s);
        return rewriteReductionStmt;
      }

      using IndexNotationRewriter::visit;

      void visit(const AssignmentNode* op) {
        // Sum every reduction variables over each term
        stmt = makeReductionNotation(Assignment(op));
      }

      void visit(const WhereNode* op) {
        stmt = Where(op);
      }

      void visit(const AccelerateNode* op) {
        stmt = Accelerate(op);
      }

    };

    return MakeReductionNotation().rewriteReduction(stmtOriginal);

  }
  return makeReductionNotation(to<Assignment>(stmtOriginal));
}

// Replace other reductions with where and forall statements
struct ReplaceReductionsWithWheres : IndexNotationRewriter {
  using IndexNotationRewriter::visit;

  Reduction reduction;
  TensorVar t;

  void visit(const AssignmentNode* node) {
    reduction = Reduction();
    t = TensorVar();

    IndexExpr rhs = rewrite(node->rhs);

    // nothing was rewritten
    if (rhs == node->rhs) {
      stmt = node;
      return;
    }

    taco_iassert(t.defined() && reduction.defined());
    IndexStmt consumer = Assignment(node->lhs, rhs, node->op);
    IndexStmt producer = forall(reduction.getVar(),
                                Assignment(t, reduction.getExpr(),
                                           reduction.getOp()));
    stmt = where(rewrite(consumer), rewrite(producer));
  }

  void visit(const ReductionNode* node) {
    // only rewrite one reduction at a time
    if (reduction.defined()) {
      expr = node;
      return;
    }

    reduction = node;
    t = TensorVar("t" + util::toString(node->var),
                  node->getDataType());
    expr = t;
  }
};

static Access replaceTemporary(IndexStmt stmt, IndexExpr expr, AcceleratorAssignment assign, ArgumentMap argumentMap){

  taco_uassert(argumentMap.possible);
  map<IndexVar, Dimension> indexVarDomains = expr.getIndexVarDomains();
  std::vector<Dimension> lhsDimension; 
  std::vector<IndexVar> indexingVec;


  for (auto var: assign.getLhs().getIndexVars()){
    lhsDimension.push_back(indexVarDomains[argumentMap.indexVars[var]]);
    indexingVec.push_back(argumentMap.indexVars[var]);
  }

  TensorVar t = TensorVar(Type(expr.getDataType(), Shape(lhsDimension)));
  return Access(t, indexingVec);
}


static Assignment getTensorAccess(IndexStmt stmt, TensorVar t)
{ 
  Assignment assign;
  bool tensorAccess = false;

  match(stmt,
    function<void(const AccessNode*)>([&](const AccessNode* n) {
      if (n->tensorVar.getName() == t.getName()){
        tensorAccess = true;
      }
    }),
    function<void(const AssignmentNode*,Matcher*)>([&](const AssignmentNode* n,
                                                       Matcher* ctx) {
      ctx->match(n->rhs);
      if (tensorAccess){
        assign = n;
      }
    })
  );

  return assign;
}


static Forall getForAllTensor(IndexStmt stmt, TensorVar t)
{ 
  Forall forall;
  bool tensorForall= false;

  //want the innermost forall
  // bool matched = false;

  match(stmt,
    function<void(const AccessNode*)>([&](const AccessNode* n) {
      if (n->tensorVar.getName() == t.getName()){
        tensorForall = true;
      }
    }),
    function<void(const ForallNode*,Matcher*)>([&](const ForallNode* n,
                                                       Matcher* ctx) {
      ctx->match(n->stmt);
      if (tensorForall){
        forall = n;
        // matched = true;
      }
    })
  );

  return forall;
}

// static std::set<IndexVar> getAllMatchingForallVars(IndexStmt stmt, std::vector<IndexVar> toMatch){

//   std::set<IndexVar> result;

//   match(stmt,
//         function<void(const ForallNode*,Matcher*)>([&](const ForallNode* n,
//                                                        Matcher* ctx) {

//       if (util::contains(toMatch, n->indexVar)){
//         result.insert(n->indexVar);   
//       }                                           
//       ctx->match(n->stmt);
//     })
//   );

//   return result;
// }

// static std::set<IndexVar> getAllNonMatchingForallVars(IndexStmt stmt, std::vector<IndexVar> toMatch){

//   std::set<IndexVar> result;

//   match(stmt,
//         function<void(const ForallNode*,Matcher*)>([&](const ForallNode* n,
//                                                        Matcher* ctx) {
//       if (!util::contains(toMatch, n->indexVar)){
//         result.insert(n->indexVar);   
//       }                                                  
//       ctx->match(n->stmt);
//     })
//   );

//   return result;
// }

static IndexStmt rewriteStmt(IndexStmt stmtRewrite, Access workspace, ConcreteAccelerateCodeGenerator codeGen, FunctionInterface functionInterface, ArgumentMap argumentMap){
    auto tensorAccess = getTensorAccess(stmtRewrite, workspace.getTensorVar());
    if (tensorAccess.defined()){
      IndexExpr rhs =  codeGen.getRHS();
      IndexStmt producerExpr = Assignment(workspace, rhs);
     
      std::map<IndexVar, IndexVar> precomputeMap;
      std::vector<IndexVar> precomputeVars;

      for (auto iVar : tensorAccess.getRhs().getIndexVars()){ 
        IndexVar iv;
        precomputeMap[iVar] = iv;
        precomputeVars.push_back(iv);
      }

      producerExpr = replace(producerExpr, precomputeMap);
      Assignment producerAssign = to<Assignment>(producerExpr);

      IndexStmt producer = InterfaceCall(producerAssign, codeGen, workspace.getTensorVar()); 

      string reason;
      for (int l = 0; l < (int) precomputeVars.size(); l++) {
        IndexVar i = tensorAccess.getLhs().getIndexVars().at(l);
        IndexVar iw = precomputeVars.at(l);

        if (i != iw) {
          IndexVarRel rel = IndexVarRel(new AccelerateRelNode(i, iw, codeGen));
          stmtRewrite = Transformation(AddSuchThatPredicates({rel})).apply(stmtRewrite, &reason);
          if (!stmtRewrite.defined()) {
            taco_uerror << reason;
          }
        }
    }

    AcceleratorStmt referenceStmt = functionInterface.getNode()->getStmt();
    if (!isa<AcceleratorAssignment>(referenceStmt)){
    taco_uerror << "Reference statement in function interface must be an assignemnt" << endl;
    }
    AcceleratorAssignment assign = to<AcceleratorAssignment>(referenceStmt);


    auto exprDim = codeGen.getRHS().getIndexVarDomains();
    auto pluginDim = assign.getRhs().getIndexVarDomains();

    // cout << util::join(codeGen.getRHS().getIndexVarDomains()) << endl;
    // cout << util::join(assign.getRhs().getIndexVarDomains()) << endl;
    // cout << util::join(argumentMap.indexVars) << endl;
    // cout << "Precompute map " << util::join(precomputeMap) << endl;

    std::map<IndexVar, size_t> toSplit;

    for (auto iv: argumentMap.indexVars){
      if (!pluginDim.at(iv.first).isVariable() && pluginDim.at(iv.first) !=  exprDim.at(iv.second)){
        if (pluginDim.at(iv.first).getSize() < exprDim.at(iv.second).getSize()){
          taco_uerror << "Broken" << endl;
          producer = forall(precomputeMap[iv.second], producer);
          toSplit[precomputeMap[iv.second]] = pluginDim.at(iv.first).getSize();
          // producer = producer.splitWithoutSuchThat(precomputeMap[iv.second], IndexVar(), IndexVar(), pluginDim.at(iv.first).getSize());
        }
        else{
          cout << "Size dont match and tiling is not possible" << endl;
          return stmtRewrite;
        }
      }
    }

    Forall forall = getForAllTensor(stmtRewrite, workspace.getTensorVar());

    IndexStmt consumer = makeConcreteNotation(tensorAccess);
    Accelerate accel(consumer, producer,  codeGen);
    IndexStmt accelStmt = static_cast<IndexStmt>(accel);

    stmtRewrite = replace(stmtRewrite, {{forall, accelStmt}}); 

    for (auto tiling: toSplit){
      stmtRewrite = stmtRewrite.split(tiling.first, IndexVar(), IndexVar(), tiling.second);
    }
 }
  return stmtRewrite;
}

static std::vector<Argument> getConcreteArgs(const std::vector<Argument>& abstractArgs, const ArgumentMap& argumentMap){
    std::vector<Argument> newArgs;
    for (auto arg : abstractArgs){
    switch (arg.getArgType())
    {
    case DIM:
        newArgs.push_back(new DimArg(argumentMap.indexVars.at(arg.getNode<DimArg>()->indexVar)));
        break;
    case TENSOR:
      taco_uerror << "Arguments can only use TensorObjects, tried using a TensorVar." << endl;
      break; 
    case TENSOR_OBJECT:
      newArgs.push_back(new TensorVarArg(argumentMap.tensors.at(arg.getNode<TensorObjectArg>()->t)));
      break;
    case TENSORVAR:
      taco_uerror << "Arguments can only use TensorObjects, tried using a TensorVar." << endl;
      break;
    case LITERAL:
      newArgs.push_back(arg);
      break;
    case USER_DEFINED: 
    { 
      std::vector<Argument> userDefinedArgs = getConcreteArgs(arg.getNode<TransferWithArgs>()->getArgs(), argumentMap);
      auto node = arg.getNode<TransferWithArgs>();
      if (node->getReturnStore().getArgType() == UNKNOWN){
        newArgs.push_back(new TransferWithArgs(node->getName(), node->getReturnType(), userDefinedArgs));
      }else{
        std::vector<Argument> returnStore = getConcreteArgs({node->getReturnStore()}, argumentMap);
        newArgs.push_back(new TransferWithArgs(node->getName(), node->getReturnType(), userDefinedArgs, returnStore[0]));
      }
      break;
    }
    case DECLVAR:
      newArgs.push_back(arg);
      break;
    case DIMLIST:
      newArgs.push_back(new DimList(argumentMap.tensors.at(arg.getNode<DimList>()->t)));
      break;
    case DATA_ARRAY:
      newArgs.push_back(new DataArray(argumentMap.tensors.at(arg.getNode<DataArray>()->t)));
      break;
    case STRING:
      newArgs.push_back(arg);
      break;
    case DECLVAR_ADDR:
      newArgs.push_back(arg);
      break;
    case TENSOR_ADDR:
      newArgs.push_back(new AddrTensorVar(argumentMap.tensors.at(arg.getNode<AddrTensorVar>()->var)));
      break;
    case TENSOR_NAME:
      newArgs.push_back(new TensorName(argumentMap.tensors.at(arg.getNode<TensorName>()->var)));
      break;
    case CAST:
    {
      std::vector<Argument> userDefinedArgs = getConcreteArgs({arg.getNode<CastArg>()->argument}, argumentMap);
      newArgs.push_back(new CastArg(userDefinedArgs[0], arg.getNode<CastArg>()->cast));
      break;
    }
      
    default:
      cout << arg.getArgType() << endl;
      taco_uerror << "Unimplemented" << endl;
      break;
    }
  }
  return newArgs;

}

static ConcreteAccelerateCodeGenerator getConcreteCodeGenerator(IndexExpr expr, IndexExpr& workspace, ArgumentMap argumentMap, FunctionInterface functionInterface){
  assert(argumentMap.possible);
  assert(isa<Access>(workspace));
  

  AcceleratorStmt referenceStmt = functionInterface.getNode()->getStmt();
  if (!isa<AcceleratorAssignment>(referenceStmt)){
    taco_uerror << "Reference statement in function interface must be an assignemnt" << endl;
  }
  bool reference = setByReference(referenceStmt);
  AcceleratorAssignment assign = to<AcceleratorAssignment>(referenceStmt);

  if (!reference){
    argumentMap.tensors[assign.getLhs().getTensorObject()] = to<Access>(workspace).getTensorVar();
  }
  

  std::vector<Argument> newArgs = getConcreteArgs(functionInterface.getNode()->getArguments(), argumentMap);
  std::vector<Argument> callBefore = getConcreteArgs(functionInterface.getNode()->callBefore(), argumentMap);
  std::vector<Argument> callAfter = getConcreteArgs(functionInterface.getNode()->callAfter(), argumentMap);



  map<IndexVar, Dimension> indexVarDomains = expr.getIndexVarDomains();
  std::vector<Dimension> lhsDimension; 
  std::vector<IndexVar> indexingVec;

  for (auto var: assign.getLhs().getIndexVars()){
    lhsDimension.push_back(indexVarDomains[argumentMap.indexVars[var]]);
    indexingVec.push_back(argumentMap.indexVars[var]);
  }

  auto accessOriginal = to<AcceleratorAccessNode>(assign.getLhs().ptr);
  IndexExpr e;
  if (reference){
    e = static_cast<IndexExpr>(Access(argumentMap.tensors[accessOriginal->tensorObject], indexingVec));
  }else{
    e = workspace;
  }

  ConcreteAccelerateCodeGenerator concreteCodeGen = ConcreteAccelerateCodeGenerator(functionInterface.getNode()->getFunctionName(), functionInterface.getNode()->getReturnType(), e, expr, newArgs, callBefore, callAfter);

  return concreteCodeGen;
}

static std::map<IndexExpr, IndexExpr> toMatchVars(IndexExpr exprToAccelerate, std::vector<IndexVar> indexVarsToHoldConstant){

  std::map<IndexExpr, IndexExpr> tensorVarToIndexVar;
  std::map<IndexVar, Dimension> indexVarDomains = exprToAccelerate.getIndexVarDomains();

  match(exprToAccelerate,
    function<void(const AccessNode*, Matcher*)>([&](const AccessNode* op, Matcher* ctx) {
      std::vector<IndexVar> iVars;
      std::vector<Dimension> dims;
      for (auto &i : op->indexVars) {
          if (!util::contains(indexVarsToHoldConstant, i)){
            iVars.push_back(i);
            dims.push_back(indexVarDomains.at(i));
          }
      }
      TensorVar tVar(Type(op->tensorVar.getType().getDataType(), dims));
      tensorVarToIndexVar.insert({Access(op), Access(tVar, iVars)});
    })
  );
  return tensorVarToIndexVar; 

}

static IndexStmt constructInnerForalls(IndexExpr e, std::vector<IndexVar> indexVarsToHoldConstant, std::map<IndexExpr, IndexExpr> constructMap, IndexExpr toAccelerate){

  IndexStmt s; 
  std::set<IndexVar> iVars;
  match(e,
    function<void(const AccessNode*, Matcher*)>([&](const AccessNode* op, Matcher* ctx) {
      for (auto &i : op->indexVars) {
          if (!util::contains(indexVarsToHoldConstant, i)){
            iVars.insert(i);
          }
      }
    })
  );

  // std::vector<IndexStmt> stmts;

  // for (auto &entry: constructMap){
  //   Access a = to<Access>(entry.first);
  //   Access b = to<Access>(entry.second);

  //   stmts.push_back(Assignment(b, a));
  // }


  std::vector<TensorVar> tensorVars;
  match(toAccelerate,
    function<void(const AccessNode*, Matcher*)>([&](const AccessNode* op, Matcher* ctx) {
          if (!util::contains(tensorVars, op->tensorVar)){
            tensorVars.push_back(op->tensorVar);
      }
    })
  );

  std::vector<IndexStmt> stmts;

  for (auto &tvAccess: tensorVars){
    for (const auto &entry : constructMap){
      Access a = to<Access>(entry.first);
      if (a.getTensorVar().getName() == tvAccess.getName()){
        Access b = to<Access>(entry.second);
        stmts.push_back(Assignment(b, a));
      }
    }
  }
 
  int i = 0;
  for (auto iVar: iVars){
    if (i==0){
      s = forall(iVar, ForallMany(iVar, stmts));
    }else{
      s = forall(iVar, s);
    }
    i++;
  }

  return s;

}

static Access constructResultAccess(ArgumentMap argumentMap, IndexExpr exprToAccelerate, FunctionInterface functionInterface){

  AcceleratorAssignment assign = to<AcceleratorAssignment>(functionInterface.getNode()->getStmt());
  std::vector<IndexVar> iVars= assign.getLhs().getIndexVars();

  std::map<IndexVar, Dimension> indexVarDomains = exprToAccelerate.getIndexVarDomains();
  std::vector<Dimension> dims; 
  std::vector<IndexVar> indexingVars; 

  IndexVar concreteVar;
  for (auto &iVar: iVars){
    assert(argumentMap.indexVars.count(iVar));
    concreteVar = argumentMap.indexVars.at(iVar);
    indexingVars.push_back(concreteVar);
    assert(indexVarDomains.count(concreteVar));
    dims.push_back(indexVarDomains.at(concreteVar));
    
  }

  TensorVar tVar(Type(assign.getLhs().getTensorObject().getType().getDataType(), dims));
  return Access(tVar, indexingVars);

}

static IndexStmt constructProducer(Access workspace, Access result, std::vector<IndexVar> indexVarsToHoldConstant){

  std::vector<IndexVar> varsToGenerate;
  for (const auto &var: workspace.getIndexVars()){
    if (!util::contains(indexVarsToHoldConstant, var) && !util::contains(varsToGenerate, var)){ 
      varsToGenerate.push_back(var);
    }
  }

  for (const auto &var: result.getIndexVars()){
    if (!util::contains(indexVarsToHoldConstant, var) && !util::contains(varsToGenerate, var)){ 
      varsToGenerate.push_back(var);
    }
  }

  IndexStmt s = Assignment(workspace, result, Add());
  for (const auto &iVar: varsToGenerate){
    s = forall(iVar, s);
  }

  return s;

}

static std::map<IndexExpr, IndexExpr> constructTiledVars(IndexExpr exprToAccelerate, std::map<IndexVar, int> varTilings, std::map<IndexVar, IndexVar> innerVarMapping){

  std::map<IndexExpr, IndexExpr> tensorVarToIndexVar;
  std::map<IndexVar, Dimension> indexVarDomains = exprToAccelerate.getIndexVarDomains();

  match(exprToAccelerate,
    function<void(const AccessNode*, Matcher*)>([&](const AccessNode* op, Matcher* ctx) {
      std::vector<IndexVar> iVars;
      std::vector<Dimension> dims;
      for (auto &i : op->indexVars) {
          if (varTilings.count(i)){
            taco_uassert(innerVarMapping.count(i));
            iVars.push_back(innerVarMapping[i]);
            dims.push_back(Dimension(varTilings[i]));
          }else{
            iVars.push_back(i);
            dims.push_back(indexVarDomains[i]);
          }
      }
      TensorVar tVar(Type(op->tensorVar.getType().getDataType(), dims));
      tensorVarToIndexVar.insert({Access(op), Access(tVar, iVars)});
    })
  );
  return tensorVarToIndexVar; 

}


static std::vector<IndexStmt> makeInnerAssigns(IndexExpr toAccelerate, std::map<IndexExpr, IndexExpr> constructMap){
  std::vector<TensorVar> tensorVars;
  match(toAccelerate,
    function<void(const AccessNode*, Matcher*)>([&](const AccessNode* op, Matcher* ctx) {
          if (!util::contains(tensorVars, op->tensorVar)){
            tensorVars.push_back(op->tensorVar);
      }
    })
  );

  std::vector<IndexStmt> stmts;
  for (auto &tvAccess: tensorVars){
    for (const auto &entry : constructMap){
      Access a = to<Access>(entry.first);
      if (a.getTensorVar().getName() == tvAccess.getName()){
        Access b = to<Access>(entry.second);
        stmts.push_back(Assignment(b, a));
      }
    }
  }
  return stmts;
}

static Access constructTiledResult(FunctionInterface functionInterface, std::map<IndexVar, IndexVar> innerVarMapping, IndexExpr exprToAccelerate, std::map<IndexVar, int> varTilings, ArgumentMap argMap){

  std::vector<IndexVar> iVars;
  std::vector<Dimension> dims;
  std::map<IndexVar, Dimension> indexVarDomains = exprToAccelerate.getIndexVarDomains();

  AcceleratorStmt referenceStmt = functionInterface.getNode()->getStmt();
  AcceleratorAssignment assign = to<AcceleratorAssignment>(referenceStmt);
      
  for (auto &var : assign.getLhs().getIndexVars()) {
    taco_uassert(argMap.indexVars.count(var));
    auto i = argMap.indexVars.at(var);
    if (!innerVarMapping.count(i)){
      iVars.push_back(i);
      dims.push_back(indexVarDomains[i]);
    }else{
      iVars.push_back(innerVarMapping[i]);
      dims.push_back(Dimension(varTilings[i]));
    }
  }
  TensorVar tVar(Type(assign.getLhs().getTensorObject().getType().getDataType(), dims));
  return Access(tVar, iVars);
}

IndexStmt IndexStmt::tile(FunctionInterface functionInterface, IndexExpr exprToAccelerate, std::map<IndexVar, int> varTilings) const{

  AcceleratorStmt referenceStmt = functionInterface.getNode()->getStmt();
  if (!isa<AcceleratorAssignment>(referenceStmt)){
    taco_uerror << "Reference statement in function interface must be an assignment" << endl;
  }

  AcceleratorAssignment assign = to<AcceleratorAssignment>(referenceStmt);
  ArgumentMap argumentMap = hasPreciseMatch(exprToAccelerate, assign.getRhs());

  if (!argumentMap.possible){
    taco_uerror << "Operator Patterns for " << assign.getRhs() << " and " << exprToAccelerate << " do not match.";
  }

  std::vector<TensorVar> temps;
  // First, we separate our computation by adding a precompute
  Access result = constructResultAccess(argumentMap, exprToAccelerate, functionInterface);
  IndexStmt consumer = replace(*this, {{exprToAccelerate, result}});
  temps.push_back(result.getTensorVar());
  
  // we change our IndexVars for another var, so that our split does not split these vars
  map<IndexVar, IndexVar> replaceIndexVars;
  match(consumer,
    std::function<void(const AccessNode*)>([&](const AccessNode* op) {
      for (auto& var : op->indexVars) {
        if (!util::contains(replaceIndexVars, var)) {
          replaceIndexVars[var]  = IndexVar();
        }
      }
    })
  );

  consumer = replace(consumer, replaceIndexVars);
  auto tensorAccess = getTensorAccess(consumer, result.getTensorVar());
  Forall forall = getForAllTensor(consumer, result.getTensorVar());
  consumer = makeConcreteNotation(makeReductionNotation(tensorAccess));

  IndexStmt rewritten = Assignment(result, exprToAccelerate, Add());
  std::vector<IndexVar> reductionVars = getReductionVars(rewritten);

  IndexStmt tiledCode =  rewritten;
  std::vector<IndexVar> innerVars;
  std::vector<IndexVar> outerVars;
  std::vector<IndexVar> originalVars;

  std::map<IndexVar, IndexVar> innerVarMapping;

  IndexVar innerVar;
  IndexVar outerVar;
  for (auto reductionVar : reductionVars){
    if (util::contains(varTilings, reductionVar)){
      innerVar = IndexVar();
      outerVar = IndexVar();
      innerVarMapping[reductionVar] = innerVar;
      innerVars.push_back(innerVar);
      outerVars.push_back(outerVar);
      originalVars.push_back(reductionVar);
      tiledCode = Forall(innerVar, tiledCode);
    }else{
      tiledCode = Forall(reductionVar, tiledCode);
    }
  }

  for (auto var : rewritten.getIndexVars()){
    if (util::contains(reductionVars, var)){
      continue;
    }
    if (util::contains(varTilings, var)){
      innerVar = IndexVar();
      outerVar = IndexVar();
      innerVarMapping[var] = innerVar;
      innerVars.push_back(innerVar);
      outerVars.push_back(outerVar);
      originalVars.push_back(var);
      tiledCode = Forall(innerVar, tiledCode);

    }else{
      tiledCode = Forall(var, tiledCode);
    }
  }

  auto tensorAssigns  = constructTiledVars(exprToAccelerate, varTilings, innerVarMapping);
  // Assignment(replace(exprToAccelerate, tensorAssigns));
  IndexStmt constructTemps = ForallMany(makeInnerAssigns(exprToAccelerate, tensorAssigns));
  tiledCode = replace(tiledCode, {{rewritten, constructTemps}});

  Access interfaceResult = constructTiledResult(functionInterface, innerVarMapping, exprToAccelerate, varTilings, argumentMap);
  temps.push_back(interfaceResult.getTensorVar());
  Assignment replacedWithTiled =  to<Assignment>(replace(Assignment(interfaceResult, replace(exprToAccelerate, tensorAssigns)), innerVarMapping));  
  IndexExpr resExpr = replacedWithTiled.getLhs();
  InterfaceCall call(replacedWithTiled, getConcreteCodeGenerator(replacedWithTiled.getRhs(), resExpr, hasPreciseMatch(replacedWithTiled.getRhs(), assign.getRhs()), functionInterface), interfaceResult.getTensorVar());

  IndexStmt setWorkspace = Assignment(result, resExpr, Add());

  for (auto var: resExpr.getIndexVars()){
    setWorkspace = Forall(var, setWorkspace);
  }

  IndexStmt copyTemp;
  if (setByReference(assign)){
    copyTemp = makeConcreteNotation(makeReductionNotation(Assignment(interfaceResult, call.getAccelGen().getLHS())));
  }

  if (copyTemp.defined()){
    copyTemp = ForallMany({tiledCode, copyTemp, call, setWorkspace});
  }else{
    copyTemp = ForallMany({tiledCode, call, setWorkspace});
  }
  
  tiledCode = copyTemp;
 
  for (auto var : outerVars){
    tiledCode = Forall(var, tiledCode);
  }

  for (auto &entry: tensorAssigns){
     temps.push_back(to<Access>(entry.second).getTensorVar());
  }

  tiledCode = DimReduction(consumer, tiledCode, temps);

  //now add the split rel node for the appropriate indexVars
  for (size_t i = 0; i < innerVars.size(); i++){
    tiledCode = tiledCode.splitWithoutRewrite(originalVars[i], outerVars[i], innerVars[i], varTilings[originalVars[i]]);
  }

  return tiledCode;
}

IndexStmt IndexStmt::holdConstant(FunctionInterface functionInterface, IndexExpr exprToAccelerate, std::vector<IndexVar> indexVarsToHoldConstant, Access workspace) const{

  if (indexVarsToHoldConstant.size() == 0){
    taco_uerror << "Please use the accelerate command!";
  }
  
  AcceleratorStmt referenceStmt = functionInterface.getNode()->getStmt();
  if (!isa<AcceleratorAssignment>(referenceStmt)){
    taco_uerror << "Reference statement in function interface must be an assignment" << endl;
  }
  AcceleratorAssignment assign = to<AcceleratorAssignment>(referenceStmt);

  std::map<IndexExpr, IndexExpr> tensorVarToIndexVar = toMatchVars(exprToAccelerate, indexVarsToHoldConstant);
  IndexExpr e = replace(exprToAccelerate, tensorVarToIndexVar);

  ArgumentMap argumentMap = hasPreciseMatch(e, assign.getRhs());

  if (!argumentMap.possible){
    taco_uerror << "Expressions " << assign.getRhs() << " and " << exprToAccelerate << " do not match when " 
      << util::join(indexVarsToHoldConstant) << " are held constant." << endl;
  }

  Access result = constructResultAccess(argumentMap, e, functionInterface);

  IndexStmt s = constructInnerForalls(e, indexVarsToHoldConstant, tensorVarToIndexVar, exprToAccelerate);
  IndexStmt producer = constructProducer(workspace, result, indexVarsToHoldConstant);

  InterfaceCall call(Assignment(result, e), getConcreteCodeGenerator(e, result, argumentMap, functionInterface), result.getTensorVar());

  IndexStmt reducedCode;

  IndexStmt copyTemp;
  if (setByReference(assign)){
    copyTemp = makeConcreteNotation(makeReductionNotation(Assignment(result, call.getAccelGen().getLHS())));
  }


  int i = 0;
  for (const auto &constantVar : indexVarsToHoldConstant){
    if (i == 0){
      if (copyTemp.defined()){
        reducedCode = forall(constantVar, ForallMany(constantVar, {s, copyTemp, call, producer}));
      }else{
        reducedCode = forall(constantVar, ForallMany(constantVar, {s, call, producer}));
      } 
    }else{
      reducedCode = forall(constantVar, reducedCode);
    }
    i++;

  }

  std::map<IndexExpr, IndexExpr> substitution;
  substitution[exprToAccelerate] = workspace;

  IndexStmt rewritten = replace(*this, substitution);

  auto tensorAccess = getTensorAccess(rewritten, workspace.getTensorVar());
  
  Forall forall = getForAllTensor(rewritten, workspace.getTensorVar());
  IndexStmt consumer = makeConcreteNotation(tensorAccess);
  rewritten = replace(rewritten, {{forall, consumer}});

  std::vector<TensorVar> temps;
  temps.push_back(workspace.getTensorVar());
  temps.push_back(result.getTensorVar());

  for (auto &entry: tensorVarToIndexVar){
     temps.push_back(to<Access>(entry.second).getTensorVar());
  }
  // return DimReduction(rewritten, reducedCode, temps);

  return DimReduction(rewritten, reducedCode, temps);
}


IndexStmt IndexStmt::accelerate(FunctionInterface functionInterface, IndexExpr exprToAccelerate, bool fullStmt) const{


  AcceleratorStmt referenceStmt = functionInterface.getNode()->getStmt();
  if (!isa<AcceleratorAssignment>(referenceStmt)){
    taco_uerror << "Reference statement in function interface must be an assignment" << endl;
  }
  AcceleratorAssignment assign = to<AcceleratorAssignment>(referenceStmt);
  // explictly add the reduction nodes
  AcceleratorAssignment assignRedux = makeReductionNotation(assign);

  ArgumentMap argumentMap;
  AcceleratorExpr rhs = assignRedux.getRhs();
  argumentMap = hasPreciseMatch(exprToAccelerate, rhs);
  if (!argumentMap.possible){
    // get reduction notation for assignment using exprToAccelerate and then check
    argumentMap = hasPreciseMatch(exprToAccelerate, assign.getRhs());
    if (argumentMap.possible){
       std::cout << "Warning : Implicit Reduction is being added, given function is calculating " << assignRedux.getRhs() << "." << std::endl;
       // Generate code to check against SMT query.
        if (functionInterface.getNode()->getConstraints().defined()){
          std::map<IndexVar, int> currentDims;
          for (auto entry : exprToAccelerate.getIndexVarDomains()){
            currentDims[entry.first] = (int) entry.second.getSize();
          }
          GenerateSMTCode condition(functionInterface.getNode()->getConstraints(), {}, currentDims, true);
          // If we cannot satisfy query even with tilings, skip.
          if (!condition.isSat()){
            taco_uerror << "Cannot satify dynamic constraints" << endl;
          };
        }
    }else{
      taco_uerror << "Expressions " << assign.getRhs() << " and " << exprToAccelerate << " do not match." << endl;
    }
  }

  assert(argumentMap.possible);

  if (fullStmt){
    // we are replacing the full statement, this means that we do not to insert a 
    // temp worksparse
    const AccessNode * lhsAccess;
    match(*this,
    function<void(const AccessNode*,Matcher*)>([&](const AccessNode* n,
                                                       Matcher* ctx) {
      lhsAccess = n;      
    }),
    function<void(const AssignmentNode*,Matcher*)>([&](const AssignmentNode* n,
                                                       Matcher* ctx) {
      if (!equals(n->rhs, exprToAccelerate)){
        taco_uerror << "Indicated that " << exprToAccelerate << "is the full expression, but not true";
      }
      ctx->match(n->lhs);
      })
    );

    std::vector<IndexStmt> stmts;
    auto access  = Access(lhsAccess);
    if (setByReference(referenceStmt)){
      std::vector<IndexVar> vars; 
      TensorVar copyTemp = argumentMap.tensors[assign.getLhs().getTensorObject()];
      if (access.getTensorVar().getOrder() != copyTemp.getOrder()){
        taco_uerror << "Set by reference, but of different orders??";
      }
      for (int i = 0; i < copyTemp.getOrder(); i++){
        vars.push_back(IndexVar());
      }
      stmts.push_back(makeConcreteNotation(Assignment(Access(access.getTensorVar(), vars), Access(copyTemp, vars))));
    }
    
    stmts.push_back(InterfaceCall(Assignment(Access(lhsAccess), exprToAccelerate), getConcreteCodeGenerator(exprToAccelerate, access, argumentMap, functionInterface), access.getTensorVar()));
    return ForallMany(stmts);
  }


  auto access = replaceTemporary(*this, exprToAccelerate, assign, argumentMap);
  std::map<IndexExpr,IndexExpr> subsitution = {{exprToAccelerate, access}};
  IndexStmt stmt =  replace(*this, subsitution);

  stmt = rewriteStmt(stmt, access, getConcreteCodeGenerator(exprToAccelerate, access, argumentMap, functionInterface), functionInterface, argumentMap);

  return stmt; 
}

IndexExpr IndexStmt::tryIndicesConstant(AcceleratorExpr toMatch, IndexExpr expr, bool& success) const {
  
  // The operator pattern of toMatch and expr must be the same.
  for (auto matchingTensor: getMatchingTensors(expr, toMatch)){
    if (isa<Access>(matchingTensor.first) && isa<AcceleratorAccess>(matchingTensor.second)){
      Access access = to<Access>(matchingTensor.first);
      AcceleratorAccess accelAccess = to<AcceleratorAccess>(matchingTensor.second);

      if (access.getTensorVar().getOrder() <= accelAccess.getTensorObject().getOrder()){
        success = false;
        return IndexExpr();
      }
  
    }
    if (isa<Literal>(matchingTensor.first) && isa<AcceleratorAccess>(matchingTensor.second)){
      success = false;
      return IndexExpr();
    }
  }

  success = true;
  return expr;

}

IndexExpr IndexStmt::tryPromotion(AcceleratorExpr toMatch, IndexExpr expr, bool& success) const{

  // The operator pattern of toMatch and expr must be the same.
  for (auto matchingTensor: getMatchingTensors(expr, toMatch)){
    if (isa<Access>(matchingTensor.first) && isa<AcceleratorAccess>(matchingTensor.second)){
      Access access = to<Access>(matchingTensor.first);
      AcceleratorAccess accelAccess = to<AcceleratorAccess>(matchingTensor.second);

      if (access.getTensorVar().getOrder() >= accelAccess.getTensorObject().getOrder()){
        success = false;
        return IndexExpr();
      }
  
    }
    if (isa<Access>(matchingTensor.first) && isa<AcceleratorLiteral>(matchingTensor.second)){
      success = false;
      return IndexExpr();
    }
  }
  success = true;
  return expr;
}


std::vector<IndexStmt> generateEquivalentStmts(IndexStmt stmt){
  std::map<IndexExpr, std::vector<IndexExpr>> exprToreplace;
  std::vector<IndexStmt>  possibleRewrites;
  
  addIdentityRewrite(stmt, exprToreplace);
  addCommutativityRewrite(stmt, exprToreplace);
  addDistributivityRewrites(stmt, exprToreplace);
  takeCommonTermsOut(stmt, exprToreplace); 


  for (auto const& it : exprToreplace){
    for (auto expr: it.second){
      std::map<IndexExpr, IndexExpr> substitution;
      substitution[it.first] = expr;
      possibleRewrites.push_back(replace(stmt, substitution));
    }
  }

  return possibleRewrites;

}


std::vector<IndexStmt> generateEquivalentStmts(IndexStmt stmt, bool distinguish, bool identity){
  std::map<IndexExpr, std::vector<IndexExpr>> exprToreplace;
  std::vector<IndexStmt>  possibleRewrites;
  // if (identity){
    addIdentityRewrite(stmt, exprToreplace);
  // }
  addCommutativityRewrite(stmt, exprToreplace);
  addDistributivityRewrites(stmt, exprToreplace);
  takeCommonTermsOut(stmt, exprToreplace); 


  for (auto const& it : exprToreplace){
    for (auto expr: it.second){
      std::map<IndexExpr, IndexExpr> substitution;
      substitution[it.first] = expr;
      possibleRewrites.push_back(replace(stmt, substitution));
    }
  }

  return possibleRewrites;

}

std::vector<IndexStmt> generateEquivalentStmts(IndexStmt stmt, int depth){

  
  std::vector<IndexStmt>  possibleRewrites = {stmt};
  int indexRewritesNotExplored = 0;  
  // generate possible rewrites upto a given depth
  for (int j = 0; j < depth; j++){
    int current_size = possibleRewrites.size();
    for (; indexRewritesNotExplored < current_size; indexRewritesNotExplored++){
      std::vector<IndexStmt> rewritesGenerated;
      if (depth <= 2){
        rewritesGenerated = generateEquivalentStmts(possibleRewrites[indexRewritesNotExplored], true, true);
      }
      else{
       rewritesGenerated = generateEquivalentStmts(possibleRewrites[indexRewritesNotExplored], true, false);
      }
      possibleRewrites.insert(possibleRewrites.end(), rewritesGenerated.begin(), rewritesGenerated.end());
    }
   
  }

  return possibleRewrites;
}

static void makeCombiUtil(vector<vector<int> >& ans,
    vector<int>& tmp, int n, int left, int k)
{
    // Pushing this vector to a vector of vector
    if (k == 0) {
        ans.push_back(tmp);
        return;
    }
 
    // i iterates from left to n. First time
    // left will be 1
    for (int i = left; i <= n; ++i)
    {
        tmp.push_back(i);
        makeCombiUtil(ans, tmp, n, i + 1, k - 1);
 
        // Popping out last inserted element
        // from the vector
        tmp.pop_back();
    }
}
 
// Prints all combinations of size k of numbers
// from 1 to n.
static vector<vector<int> > makeCombi(int n, int k)
{
    vector<vector<int> > ans;
    vector<int> tmp;
    makeCombiUtil(ans, tmp, n, 1, k);
    return ans;
}

IndexStmt IndexStmt::helperCheckForMatches(IndexStmt stmt, std::vector<FunctionInterface> functionInterfaces, std::set<std::pair<std::string, std::string>>& expressions) const{

  std::stack<std::tuple<Access, ConcreteAccelerateCodeGenerator, FunctionInterface, ArgumentMap>> varCodeGen;
  // std::map<ConcreteAccelerateCodeGenerator, FunctionInterface> abstractInterface;

  if (!isa<Assignment>(stmt)) {
    cout << "Cannot autoschedule this expression since it is not an assignment" << endl;
    return stmt;
  }

  IndexStmt stmtRewrite = stmt;
  for (auto descripton: functionInterfaces){
    AcceleratorStmt referenceStmt = descripton.getNode()->getStmt();
    
    if (!isa<AcceleratorAssignment>(referenceStmt)){
      taco_uerror << "Reference statement in function interface must be an assignemnt" << endl;
    }

    AcceleratorAssignment assign = to<AcceleratorAssignment>(referenceStmt);
    AcceleratorAssignment reduxRefStmt = makeReductionNotation(assign);
    std::vector<IndexExpr> matchedExprs = allMatchedOpPatterns(to<Assignment>(stmt).getRhs(), reduxRefStmt.getRhs());
    ArgumentMap argumentMap;

    for (auto expr: matchedExprs){
      
      stringstream ss; 
      ss << expr; 

      if (expressions.count({ss.str(), descripton.getNode()->getFunctionName()})) continue;
      argumentMap = hasPreciseMatch(expr, reduxRefStmt.getRhs());
      if (argumentMap.possible){
        expressions.insert({ss.str(), descripton.getNode()->getFunctionName()});
        // Generate STMT query if a constraint exists
        // True indicates that we are interested in finding tilings.
        if (descripton.getNode()->getConstraints().defined()){
          std::map<IndexVar, int> currentDims;
          for (auto entry : expr.getIndexVarDomains()){
            currentDims[argumentMap.indexVars[entry.first]] = (int) entry.second.getSize();
          }
          // std::cout << "Check" << util::join(currentDims) << std::endl;
          GenerateSMTCode condition(descripton.getNode()->getConstraints(), {}, currentDims, true);

          // If we cannot satisfy query even with tilings, skip.
          if (!condition.isSat()){
            continue;
          }
        }

        // std::cout << descripton.getNode()->getFunctionName() << std::endl;
        auto access = replaceTemporary(stmt, expr, reduxRefStmt, argumentMap);
        std::map<IndexExpr,IndexExpr> subsitution = {{expr, access}};
        stmtRewrite =  replace(stmtRewrite, subsitution);
        auto codeGen = getConcreteCodeGenerator(expr, access, argumentMap, descripton);
        // varCodeGen.push(std::make_tuple(access, codeGen, descripton, argumentMap));
        // // break;
        }else{
          std::vector<IndexVar> allVars = taco::getIndexVars(expr);
          // expressions.insert({ss.str(), descripton.getNode()->getFunctionName()});
          bool found = false;

          for (int i = 0; i < allVars.size(); i++){
            // std::cout << "starting" << std::endl;
            // We want to stop when we have found the minimum number of indices
            // to hold constant.
            auto samples = makeCombi(allVars.size(), i);
            for (auto sample: samples){
              std::vector<IndexVar> holdConstant;
              for (auto s: sample){
                holdConstant.push_back(allVars[s-1]);
              }
              // std::cout << util::join(holdConstant) << std::endl;
              auto tensorVarsnew = toMatchVars(expr, holdConstant);
              IndexExpr e = replace(expr, tensorVarsnew);

              // std::cout << e << std::endl;
              ArgumentMap argumentMapConst = hasPreciseMatch(e, reduxRefStmt.getRhs());
              if (argumentMapConst.possible){
                found = true;
                expressions.insert({ss.str(), descripton.getNode()->getFunctionName()});

                if (descripton.getNode()->getConstraints().defined()){
                  std::map<IndexVar, int> currentDims;
                  for (auto entry : e.getIndexVarDomains()){
                    currentDims[argumentMap.indexVars[entry.first]] = (int) entry.second.getSize();
                  }
                  // std::cout << "Check" << util::join(currentDims) << std::endl;
                  GenerateSMTCode condition(descripton.getNode()->getConstraints(), {}, currentDims, true);

                  // If we cannot satisfy query even with tilings, skip.
                  if (!condition.isSat()){
                    continue;
                  }
                }
               }
             }
          if (found){
            break;
          }
         }
        }
      }
  }
  
  stmtRewrite = makeConcreteNotation(stmtRewrite);

  while (!varCodeGen.empty()){
    auto tensorCodeGen = varCodeGen.top();
    stmtRewrite = rewriteStmt(stmtRewrite, std::get<0>(tensorCodeGen), std::get<1>(tensorCodeGen), std::get<2>(tensorCodeGen), std::get<3>(tensorCodeGen));
    varCodeGen.pop();
  }

  return stmtRewrite;
}

std::vector<IndexStmt> IndexStmt::autoAccelerate(IndexStmt stmt, std::vector<FunctionInterface> functionInterfaces) const{

  auto start1 = std::chrono::high_resolution_clock::now();

  std::vector<IndexStmt> possibleRewrites = generateEquivalentStmts(stmt, 3);
  std::vector<IndexStmt> possibleStmts;
  possibleStmts.push_back(makeConcreteNotation(stmt));

  std::set<std::pair<std::string, std::string>> expressions;
  // Account for the case where there are no mappings.
  expressions.insert({"", ""});
  helperCheckForMatches(stmt, functionInterfaces, expressions);

  for (int i = 0; i < possibleRewrites.size(); i++){
      possibleStmts.push_back(helperCheckForMatches(possibleRewrites[i], functionInterfaces, expressions));
  }


  // std::cout << " check ";
  auto end1 = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);

  std::cout << "Time taken to Run Completely: "
        << float(duration.count())/float(1000000) << "s" << std::endl;

  std::cout << "Total Number: "
        << expressions.size() << std::endl;
  for (auto mappings: expressions){
    std::cout << mappings.first << " " << mappings.second << std::endl;
  }



  return possibleStmts;
}


IndexStmt makeConcreteNotation(IndexStmt stmt) {

  std::string reason;
  taco_iassert(isReductionNotation(stmt, &reason))
      << "Not reduction notation: " << stmt << std::endl << reason;
  taco_iassert(isa<Assignment>(stmt));

  // Free variables and reductions covering the whole rhs become top level loops
  vector<IndexVar> freeVars = to<Assignment>(stmt).getFreeVars();

  struct RemoveTopLevelReductions : IndexNotationRewriter {
    using IndexNotationRewriter::visit;
    //take no action in an accelerate node
    void visit(const AccelerateNode* node) {
      return;
    }
    void visit(const AssignmentNode* node) {
      // Easiest to just walk down the reduction node until we find something
      // that's not a reduction
      // cout << "assignent node" << *node << endl;
      vector<IndexVar> topLevelReductions;
      IndexExpr rhs = node->rhs;
      IndexExpr reductionOp;
      while (isa<Reduction>(rhs)) {
        Reduction reduction = to<Reduction>(rhs);
        // Hack: explicit reductions with user defined functions shouldn't be rewritten.
        if (util::getFromEnv("TACO_CONCRETIZE_HACK", "0") != "0" && isa<Call>(reduction.getOp())) {
          break;
        }
        topLevelReductions.push_back(reduction.getVar());
        rhs = reduction.getExpr();
        reductionOp = reduction.getOp();
      }

      if (rhs != node->rhs) {
        stmt = Assignment(node->lhs, rhs, reductionOp);
        for (auto& i : util::reverse(topLevelReductions)) {
          stmt = forall(i, stmt);
        }
      }
      else {
        stmt = node;
      }
    }
  };

  stmt = RemoveTopLevelReductions().rewrite(stmt);


  for (auto& i : util::reverse(freeVars)) {
    stmt = forall(i, stmt);
  }

  // Replace other reductions with where and forall statements
  struct ReplaceReductionsWithWheres : IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    Reduction reduction;
    TensorVar t;

    void visit(const AssignmentNode* node) {
      reduction = Reduction();
      t = TensorVar();

      IndexExpr rhs = rewrite(node->rhs);

      // nothing was rewritten
      if (rhs == node->rhs) {
        stmt = node;
        return;
      }

      taco_iassert(t.defined() && reduction.defined());
      IndexStmt consumer = Assignment(node->lhs, rhs, node->op);
      IndexStmt producer = forall(reduction.getVar(),
                                  Assignment(t, reduction.getExpr(),
                                             reduction.getOp()));
      stmt = where(rewrite(consumer), rewrite(producer));
    }

    void visit(const ReductionNode* node) {
      // only rewrite one reduction at a time
      if (reduction.defined()) {
        expr = node;
        return;
      }

      reduction = node;
      t = TensorVar("t" + util::toString(node->var),
                    node->getDataType());
      expr = t;
    }
  };
  stmt = ReplaceReductionsWithWheres().rewrite(stmt);

  return stmt;
}

Assignment makeReductionNotationScheduled(Assignment assignment, ProvenanceGraph provGraph) {
  IndexExpr expr = assignment.getRhs();
  std::vector<IndexVar> free = assignment.getLhs().getIndexVars();
  if (!isEinsumNotation(assignment)) {
    return assignment;
  }

  struct MakeReductionNotation : IndexNotationRewriter {
    MakeReductionNotation(const std::vector<IndexVar>& free, ProvenanceGraph provGraph)
      : free(free.begin(), free.end()), provGraph(provGraph){}

    std::set<IndexVar> free;
    ProvenanceGraph provGraph; 
    bool onlyOneTerm;

    IndexExpr addReductions(IndexExpr expr) {
      auto vars = getIndexVars(expr);
      for (auto& var : util::reverse(vars)) {

        if (!util::contains(free, var)) {
          bool shouldReduce = true;
          /// Do not add a reduction node if mismatch is between a fully derived indexVar and its ancestor
          if (provGraph.isFullyDerived(var)) {
            for (auto& f: free) {
              if (provGraph.isDerivedFrom(var, f)) {
                shouldReduce = false;
              }
            }
          } else {
            for (auto& f: free) {
              if (provGraph.isDerivedFrom(f, var)) {
                shouldReduce = false;
              }
            }
          }
          if (shouldReduce)
            expr = sum(var,expr);
        }
      }
      return expr;
    }

    IndexExpr einsum(const IndexExpr& expr) {
      onlyOneTerm = true;
      IndexExpr einsumexpr = rewrite(expr);

      if (onlyOneTerm) {
        einsumexpr = addReductions(einsumexpr);
      }

      return einsumexpr;
    }

    using IndexNotationRewriter::visit;

    void visit(const AddNode* op) {
      // Sum every reduction variables over each term
      onlyOneTerm = false;

      IndexExpr a = addReductions(op->a);
      IndexExpr b = addReductions(op->b);
      if (a == op->a && b == op->b) {
        expr = op;
      }
      else {
        expr = new AddNode(a, b);
      }
    }

    void visit(const SubNode* op) {
      // Sum every reduction variables over each term
      onlyOneTerm = false;

      IndexExpr a = addReductions(op->a);
      IndexExpr b = addReductions(op->b);
      if (a == op->a && b == op->b) {
        expr = op;
      }
      else {
        expr = new SubNode(a, b);
      }
    }
  };
  return Assignment(assignment.getLhs(),
                    MakeReductionNotation(free, provGraph).einsum(expr),
                    assignment.getOperator());
}

IndexStmt makeReductionNotationScheduled(IndexStmt stmt, ProvenanceGraph provGraph) {
  taco_iassert(isEinsumNotation(stmt));
  return makeReductionNotationScheduled(to<Assignment>(stmt), provGraph);
}

IndexStmt makeConcreteNotationScheduled(IndexStmt stmt, ProvenanceGraph provGraph, 
                                        vector<IndexVar> forallIndexVars) {
  std::string reason;
  taco_iassert(isReductionNotationScheduled(stmt, provGraph, &reason))
    << "Not reduction notation: " << stmt << std::endl << reason;
  taco_iassert(isa<Assignment>(stmt));

  // Free variables and reductions covering the whole rhs become top level loops
  vector<IndexVar> freeVars = to<Assignment>(stmt).getFreeVars();
  vector<IndexVar> reductionAndFreeVars;

  struct RemoveTopLevelReductions : IndexNotationRewriter {
    using IndexNotationRewriter::visit;
    vector<IndexVar> forallIndexVars;
    vector<IndexVar> reductionAndFreeVars;

    RemoveTopLevelReductions(vector<IndexVar> forallIndexVars) : forallIndexVars(forallIndexVars) {}

    void visit(const AssignmentNode* node) {
      // Easiest to just walk down the reduction node until we find something
      // that's not a reduction
      vector<IndexVar> topLevelReductions;
      IndexExpr rhs = node->rhs;
      while (isa<Reduction>(rhs)) {
        Reduction reduction = to<Reduction>(rhs);
        topLevelReductions.push_back(reduction.getVar());
        rhs = reduction.getExpr();
      }

      if (rhs != node->rhs) {
        stmt = Assignment(node->lhs, rhs, Add());
        if (forallIndexVars.empty()) {
          for (auto &i : util::reverse(topLevelReductions)) {
            stmt = forall(i, stmt);
          }
        } else {
          reductionAndFreeVars.insert(reductionAndFreeVars.end(), topLevelReductions.begin(), 
                                      topLevelReductions.end());
        }
      }
      else {
        stmt = node;
      }
    }
  };
  auto rewriter = RemoveTopLevelReductions(forallIndexVars);
  stmt = rewriter.rewrite(stmt);
  reductionAndFreeVars = rewriter.reductionAndFreeVars;
  // This gets the list of indexVars on the rhs of an assignment
  // TODO: check to make sure that we want to get ALL rhs indexVars (not just the upper level)
  vector<IndexVar> rhsVars;
  match(stmt,
        function<void(const AccessNode*, Matcher*)>([&](const AccessNode* op, Matcher* ctx) {
          for (auto &i : op->indexVars) {
            if (std::find(rhsVars.begin(), rhsVars.end(), i) == rhsVars.end()) {
              rhsVars.push_back(i);
            }
          }
        }),
        function<void(const AssignmentNode*, Matcher*)>([&](const AssignmentNode* op, Matcher* ctx) {
          ctx->match(op->rhs);
        })
  );

  // Emit the freeVars as foralls if the freeVars are fully derived
  // else emit the fully derived descendant of the freeVar found in rhsVars
  if (forallIndexVars.empty()) {
    for (auto &i : util::reverse(freeVars)) {
      if (provGraph.isFullyDerived(i))
        stmt = forall(i, stmt);
      else {
        auto derivedVars = provGraph.getFullyDerivedDescendants(i);
        IndexVar derivedI = *rhsVars.begin();
        for (auto &derivedVar : derivedVars) {
          if (std::find(rhsVars.begin(), rhsVars.end(), derivedVar) != rhsVars.end()) {
            derivedI = derivedVar;
          }
        }
        stmt = forall(derivedI, stmt);
      }
    }
  } else {
    reductionAndFreeVars.insert(reductionAndFreeVars.end(), freeVars.begin(), freeVars.end());
    for (auto &i : util::reverse(forallIndexVars)) {
      if (std::find(reductionAndFreeVars.begin(), reductionAndFreeVars.end(), i) != reductionAndFreeVars.end())
        stmt = forall(i, stmt);
      else {
        auto ancestorVars = provGraph.getUnderivedAncestors(i);
        IndexVar ancestorI = *reductionAndFreeVars.begin();
        for (auto &ancestorVar : ancestorVars) {
          if (std::find(reductionAndFreeVars.begin(), reductionAndFreeVars.end(), ancestorVar) 
              != reductionAndFreeVars.end()) {
            stmt = forall(i, stmt);
          }
        }
      }
    }
  }

  stmt = ReplaceReductionsWithWheres().rewrite(stmt);
  return stmt;
}

vector<TensorVar> getResults(IndexStmt stmt) {
  vector<TensorVar> result;
  set<TensorVar> collected;
  set<std::string> names; 

  for (auto& access : getResultAccesses(stmt).first) {
    TensorVar tensor = access.getTensorVar();
    if (util::contains(names, tensor.getName())){
      continue;
    }
    taco_iassert(!util::contains(collected, tensor));
    collected.insert(tensor);
    names.insert(tensor.getName());
    result.push_back(tensor);
  }

  return result;
}


vector<TensorVar> getArguments(IndexStmt stmt) {
  vector<TensorVar> result;
  set<TensorVar> collected;

  for (auto& access : getArgumentAccesses(stmt)) {
    TensorVar tensor = access.getTensorVar();
    if (!util::contains(collected, tensor)) {
      collected.insert(tensor);
      result.push_back(tensor);
    }
    // The arguments will include any index sets on this tensor
    // argument as well.
    if (access.hasIndexSetModes()) {
      for (size_t i = 0; i < access.getIndexVars().size(); i++) {
        if (access.isModeIndexSet(i)) {
          auto t = access.getModeIndexSetTensor(i);
          if (!util::contains(collected, t)) {
            collected.insert(t);
            result.push_back(t);
          }
        }
      }
    }
  }

  return result;
}

bool allForFreeLoopsBeforeAllReductionLoops(IndexStmt stmt) {

    struct LoopOrderGetter : IndexNotationVisitor {

      std::vector<IndexVar> loopOrder;
      std::set<IndexVar> freeVars;

      using IndexNotationVisitor::visit;

      void visit(const AssignmentNode *op) {
        for (const auto &var : op->lhs.getIndexVars()) {
          freeVars.insert(var);
        }
        IndexNotationVisitor::visit(op);
      }

      void visit(const ForallNode *op) {
        loopOrder.push_back(op->indexVar);
        IndexNotationVisitor::visit(op);
      }
    };


    LoopOrderGetter getter;
    getter.visit(stmt);

    bool seenReductionVar = false;
    for (auto &var : getter.loopOrder) {
      if (util::contains(getter.freeVars, var)) {
        if (seenReductionVar) {
          // A reduction loop came before a loop over a free var
          return false;
        }
      } else {
        seenReductionVar = true;
      }
    }
    return true;
  }

std::map<Forall, Where> getTemporaryLocations(IndexStmt stmt) {
  map<Forall, Where> temporaryLocs;
  Forall f = Forall();
  match(stmt,
        function<void(const ForallNode*, Matcher*)>([&](const ForallNode* op, Matcher* ctx) {
          f = op;
          ctx->match(op->stmt);
        }),
          function<void(const WhereNode*, Matcher*)>([&](const WhereNode* w, Matcher* ctx) {
            if (!(f == IndexStmt()))
              temporaryLocs.insert({f, Where(w)});
          })
        );
  return temporaryLocs;
}


std::vector<TensorVar> getTemporaries(IndexStmt stmt) {
  vector<TensorVar> temporaries;
  bool firstAssignment = true;
  match(stmt,
    function<void(const AssignmentNode*)>([&](const AssignmentNode* op) {
      // Ignore the first assignment as its lhs is the result and not a temp.
      if (firstAssignment) {
        firstAssignment = false;
        return;
      }
      temporaries.push_back(op->lhs.getTensorVar());
    }),
    function<void(const SequenceNode*,Matcher*)>([&](const SequenceNode* op,
                                                     Matcher* ctx) {
      if (firstAssignment) {
        ctx->match(op->definition);
        firstAssignment = true;
        ctx->match(op->mutation);
      }
      else {
        ctx->match(op->definition);
        ctx->match(op->mutation);
      }
    }),
    function<void(const MultiNode*,Matcher*)>([&](const MultiNode* op,
                                                  Matcher* ctx) {
      if (firstAssignment) {
        ctx->match(op->stmt1);
        firstAssignment = true;
        ctx->match(op->stmt2);
      }
      else {
        ctx->match(op->stmt1);
        ctx->match(op->stmt2);
      }
    }),
    function<void(const WhereNode*,Matcher*)>([&](const WhereNode* op,
                                                  Matcher* ctx) {
      ctx->match(op->consumer);
      ctx->match(op->producer);
    }),
    function<void(const AccelerateNode*,Matcher*)>([&](const AccelerateNode* op,
                                                  Matcher* ctx) {
      ctx->match(op->consumer);
      ctx->match(op->producer);
    }),
    function<void(const AssembleNode*,Matcher*)>([&](const AssembleNode* op,
                                                  Matcher* ctx) {
      ctx->match(op->compute);
      if (op->queries.defined()) {
        ctx->match(op->queries);
      }
    }),
    function<void(const DimReductionNode*,Matcher*)>([&](const DimReductionNode* op,
                                                  Matcher* ctx) {

      for (const auto &temp: op->temps){
        if (!util::contains(temporaries, temp)){
          temporaries.push_back(temp);
        }
      }
      
      ctx->match(op->consumer);
      ctx->match(op->producer);
    }),
    function<void(const ForallManyNode*,Matcher*)>([&](const ForallManyNode* op,
                                                  Matcher* ctx) {

      for (const auto &stmt : op->stmts){
        ctx->match(stmt);
      }

    })
  );
  return temporaries;
}


std::vector<TensorVar> getAttrQueryResults(IndexStmt stmt) {
  std::vector<TensorVar> results;
  match(stmt,
    function<void(const AssembleNode*,Matcher*)>([&](const AssembleNode* op,
                                                  Matcher* ctx) {
      const auto queryResults = getResults(op->queries);
      results.insert(results.end(), queryResults.begin(), queryResults.end());
      if (op->queries.defined()) {
        ctx->match(op->queries);
      }
      ctx->match(op->compute);
    })
  );
  return results;
}


std::vector<TensorVar> getAssembledByUngroupedInsertion(IndexStmt stmt) {
  std::vector<TensorVar> results;
  match(stmt,
    function<void(const AssembleNode*,Matcher*)>([&](const AssembleNode* op,
                                                  Matcher* ctx) {
      for (const auto& result : op->results) {
        results.push_back(result.first);
      }
      if (op->queries.defined()) {
        ctx->match(op->queries);
      }
      ctx->match(op->compute);
    })
  );
  return results;
}


std::vector<TensorVar> getTensorVars(IndexStmt stmt) {
  vector<TensorVar> results = getResults(stmt);
  vector<TensorVar> arguments = getArguments(stmt);
  vector<TensorVar> temps = getTemporaries(stmt);
  return util::combine(results, util::combine(arguments, temps));
}


pair<vector<Access>,set<Access>> getResultAccesses(IndexStmt stmt)
{
  vector<Access> result;
  set<Access> reduced;

  match(stmt,
    function<void(const AssignmentNode*)>([&](const AssignmentNode* op) {
      taco_iassert(!util::contains(result, op->lhs));
      result.push_back(op->lhs);
      if (op->op.defined()) {
        reduced.insert(op->lhs);
      }
    }),
    function<void(const WhereNode*,Matcher*)>([&](const WhereNode* op,
                                                  Matcher* ctx) {
      ctx->match(op->consumer);
    }),
    function<void(const AccelerateNode*,Matcher*)>([&](const AccelerateNode* op,
                                                  Matcher* ctx) {
      ctx->match(op->consumer);
    }),
    function<void(const DimReductionNode*,Matcher*)>([&](const DimReductionNode* op,
                                                  Matcher* ctx) {
      ctx->match(op->consumer);
      // ctx->match(op->producer);
    }),
    function<void(const SequenceNode*,Matcher*)>([&](const SequenceNode* op,
                                                     Matcher* ctx) {
      ctx->match(op->definition);
    }),
    function<void(const AssembleNode*,Matcher*)>([&](const AssembleNode* op,
                                                     Matcher* ctx) {
      ctx->match(op->compute);
    }),
    function<void(const ForallManyNode*,Matcher*)>([&](const ForallManyNode* op,
                                                     Matcher* ctx) {

      for (const auto& stmt: op->stmts){
        ctx->match(stmt);
      }
      
    })
  );
  return {result, reduced};
}


std::vector<Access> getArgumentAccesses(IndexStmt stmt)
{
  vector<Access> result;
  set<TensorVar> temporaries = util::toSet(getTemporaries(stmt));

  match(stmt,
    function<void(const AccessNode*)>([&](const AccessNode* n) {
      if (util::contains(temporaries, n->tensorVar)) {
        return;
      }
      result.push_back(n);
    }),
    function<void(const AssignmentNode*,Matcher*)>([&](const AssignmentNode* n,
                                                       Matcher* ctx) {
      ctx->match(n->rhs);
    })
  );

  return result;
}

// Return corresponding underived indexvars
struct GetIndexVars : IndexNotationVisitor {
  GetIndexVars(ProvenanceGraph provGraph) : provGraph(provGraph) {}
  vector<IndexVar> indexVars;
  set<IndexVar> seen;
  ProvenanceGraph provGraph;

  using IndexNotationVisitor::visit;

  void add(const vector<IndexVar>& vars) {
    for (auto& var : vars) {
      std::vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(var);
      for (auto &underived : underivedAncestors) {
        if (!util::contains(seen, underived)) {
          seen.insert(underived);
          indexVars.push_back(underived);
        }
      }
    }
  }

  void visit(const ForallNode* node) {
    add({node->indexVar});
    IndexNotationVisitor::visit(node->stmt);
  }

  void visit(const AccessNode* node) {
    add(node->indexVars);
  }

  void visit(const AssignmentNode* node) {
    add(node->lhs.getIndexVars());
    IndexNotationVisitor::visit(node->rhs);
  }
};

vector<IndexVar> getIndexVars(IndexStmt stmt) {
  GetIndexVars visitor = GetIndexVars(ProvenanceGraph(stmt));
  stmt.accept(&visitor);
  return visitor.indexVars;
}


vector<IndexVar> getIndexVars(IndexExpr expr) {
  GetIndexVars visitor = GetIndexVars(ProvenanceGraph());
  expr.accept(&visitor);
  return visitor.indexVars;
}

std::vector<IndexVar> getReductionVars(IndexStmt stmt) {
  const auto provGraph = ProvenanceGraph(stmt);

  std::vector<IndexVar> reductionVars, scopedVars, producerScopedVars, 
                        consumerScopedVars;
  match(stmt,
    function<void(const ForallNode*,Matcher*)>([&](const ForallNode* op, 
                                                   Matcher* ctx) {
      const auto indexVars = provGraph.getUnderivedAncestors(op->indexVar);
      for (const auto& iv : indexVars) {
        scopedVars.push_back(iv);
      }
      ctx->match(op->stmt);
      for (size_t i = 0; i < indexVars.size(); ++i) {
        scopedVars.pop_back();
      }
    }),
    function<void(const WhereNode*,Matcher*)>([&](const WhereNode* op,
                                                  Matcher* ctx) {
      const auto oldProducerScopedVars = producerScopedVars;
      producerScopedVars = scopedVars;
      ctx->match(op->producer);
      producerScopedVars = oldProducerScopedVars;

      const auto oldConsumerScopedVars = consumerScopedVars;
      consumerScopedVars = scopedVars;
      ctx->match(op->consumer);
      consumerScopedVars = oldConsumerScopedVars;
    }),
    function<void(const AssignmentNode*)>([&](const AssignmentNode* op) {
      auto freeVars = op->lhs.getIndexVars();
      util::append(freeVars, producerScopedVars);

      auto seen = util::toSet(freeVars);
      match(op->rhs,
        std::function<void(const AccessNode*)>([&](const AccessNode* op) {
          for (const auto& var : op->indexVars) {
            if (!util::contains(seen, var)) {
              reductionVars.push_back(var);
              seen.insert(var);
            }
          }
        })
      );
      for (const auto& var : consumerScopedVars) {
        if (!util::contains(seen, var)) {
          reductionVars.push_back(var);
          seen.insert(var);
        }
      }
    })
  );
  return reductionVars;
}

vector<ir::Expr> createVars(const vector<TensorVar>& tensorVars,
                            map<TensorVar, ir::Expr>* vars, 
                            bool isParameter) {
  taco_iassert(vars != nullptr);
  vector<ir::Expr> irVars;
  for (auto& var : tensorVars) {
    ir::Expr irVar = ir::Var::make(var.getName(), var.getType().getDataType(),
                                   true, true, isParameter);
    irVars.push_back(irVar);
    vars->insert({var, irVar});
  }
  return irVars;
}

std::map<TensorVar,ir::Expr> createIRTensorVars(IndexStmt stmt)
{
  std::map<TensorVar,ir::Expr> tensorVars;

  // Create result and parameter variables
  vector<TensorVar> results = getResults(stmt);
  vector<TensorVar> arguments = getArguments(stmt);
  vector<TensorVar> temporaries = getTemporaries(stmt);

  // Create variables for index sets on result tensors.
  for (auto& access : getResultAccesses(stmt).first) {
    // Any accesses that have index sets will be added.
    if (access.hasIndexSetModes()) {
      for (size_t i = 0; i < access.getIndexVars().size(); i++) {
        if (access.isModeIndexSet(i)) {
          auto t = access.getModeIndexSetTensor(i);
          if (tensorVars.count(t) == 0) {
            ir::Expr irVar = ir::Var::make(t.getName(), t.getType().getDataType(), true, true, true);
            tensorVars.insert({t, irVar});
          }
        }
      }
    }
  }

  // Convert tensor results, arguments and temporaries to IR variables
  map<TensorVar, ir::Expr> resultVars;
  vector<ir::Expr> resultsIR = createVars(results, &resultVars);
  tensorVars.insert(resultVars.begin(), resultVars.end());
  vector<ir::Expr> argumentsIR = createVars(arguments, &tensorVars);
  vector<ir::Expr> temporariesIR = createVars(temporaries, &tensorVars);

  return tensorVars;
}

struct Zero : public IndexNotationRewriterStrict {
public:
  Zero(const set<Access>& zeroed) : zeroed(zeroed) {}

private:
  using IndexExprRewriterStrict::visit;

  set<Access> zeroed;

  /// Temporary variables whose assignment has become zero.  These are therefore
  /// zero at every access site.
  set<TensorVar> zeroedVars;

  void visit(const AccessNode* op) {
    if (util::contains(zeroed, op) ||
        util::contains(zeroedVars, op->tensorVar)) {
      expr = IndexExpr();
    }
    else {
      expr = op;
    }
  }

  void visit(const LiteralNode* op) {
    expr = op;
  }

  void visit(const IndexVarNode* op) {
    expr = op;
  }

  template <class T>
  IndexExpr visitUnaryOp(const T *op) {
    IndexExpr a = rewrite(op->a);
    if (!a.defined()) {
      return IndexExpr();
    }
    else if (a == op->a) {
      return op;
    }
    else {
      return new T(a);
    }
  }

  void visit(const NegNode* op) {
    expr = visitUnaryOp(op);
  }

  void visit(const SqrtNode* op) {
    expr = visitUnaryOp(op);
  }

  template <class T>
  IndexExpr visitDisjunctionOp(const T *op) {
    IndexExpr a = rewrite(op->a);
    IndexExpr b = rewrite(op->b);
    if (!a.defined() && !b.defined()) {
      return IndexExpr();
    }
    else if (!a.defined()) {
      return b;
    }
    else if (!b.defined()) {
      return a;
    }
    else if (a == op->a && b == op->b) {
      return op;
    }
    else {
      return new T(a, b);
    }
  }

  template <class T>
  IndexExpr visitConjunctionOp(const T *op) {
    IndexExpr a = rewrite(op->a);
    IndexExpr b = rewrite(op->b);
    if (!a.defined() || !b.defined()) {
      return IndexExpr();
    }
    else if (a == op->a && b == op->b) {
      return op;
    }
    else {
      return new T(a, b);
    }
  }

  void visit(const AddNode* op) {
    expr = visitDisjunctionOp(op);
  }

  void visit(const SubNode* op) {
    IndexExpr a = rewrite(op->a);
    IndexExpr b = rewrite(op->b);
    if (!a.defined() && !b.defined()) {
      expr = IndexExpr();
    }
    else if (!a.defined()) {
      expr = -b;
    }
    else if (!b.defined()) {
      expr = a;
    }
    else if (a == op->a && b == op->b) {
      expr = op;
    }
    else {
      expr = new SubNode(a, b);
    }
  }

  void visit(const MulNode* op) {
    expr = visitConjunctionOp(op);
  }

  void visit(const DivNode* op) {
    expr = visitConjunctionOp(op);
  }

  void visit(const CastNode* op) {
    IndexExpr a = rewrite(op->a);
    if (!a.defined()) {
      expr = IndexExpr();
    }
    else if (a == op->a) {
      expr = op;
    }
    else {
      expr = new CastNode(a, op->getDataType());
    }
  }

  void visit(const CallNode* op) {
    std::vector<IndexExpr> args;
    std::vector<IndexExpr> rewrittenArgs;
    std::vector<int> definedArgs;
    bool rewritten = false;

    Annihilator annihilator = findProperty<Annihilator>(op->properties);

    // TODO: Check exhausted default against result default
    for(int argIdx = 0; argIdx < (int) op->args.size(); ++argIdx) {
      IndexExpr arg = op->args[argIdx];
      IndexExpr rewrittenArg = rewrite(arg);
      rewrittenArgs.push_back(rewrittenArg);

      if (rewrittenArg.defined()) {
        definedArgs.push_back(argIdx);
      } else {
        // TODO: fill value instead of 0
        rewrittenArg = Literal::zero(arg.getDataType());
      }

      args.push_back(rewrittenArg);
      if (arg != rewrittenArg) {
        rewritten = true;
      }
    }

    if(annihilator.defined()) {
      IndexExpr e = annihilator.annihilates(args);
      if(e.defined()) {
        expr = e;
        return;
      }
    }

    Identity identity = findProperty<Identity>(op->properties);
    if(identity.defined()) {
      IndexExpr e = identity.simplify(args);
      if(e.defined()) {
        expr = e;
        return;
      }
    }

    if (rewritten) {
      const std::map<IndexExpr, IndexExpr> subs = util::zipToMap(op->args, rewrittenArgs);
      IterationAlgebra newAlg = replaceAlgIndexExprs(op->iterAlg, subs);
      expr = new CallNode(op->name, args, op->defaultLowerFunc, newAlg, op->properties,
                          op->regionDefinitions, definedArgs);
    }
    else {
      expr = op;
    }

  }

  void visit(const CallIntrinsicNode* op) {
    std::vector<IndexExpr> args;
    std::vector<size_t> zeroArgs;
    bool rewritten = false;
    for (size_t i = 0; i < op->args.size(); ++i) {
      IndexExpr arg = op->args[i];
      IndexExpr rewrittenArg = rewrite(arg);
      if (!rewrittenArg.defined()) {
        rewrittenArg = Literal::zero(arg.getDataType());
        zeroArgs.push_back(i);
      }
      args.push_back(rewrittenArg);
      if (arg != rewrittenArg) {
        rewritten = true;
      }
    }
    const auto zeroPreservingArgsSets = op->func->zeroPreservingArgs(args);
    for (const auto& zeroPreservingArgs : zeroPreservingArgsSets) {
      taco_iassert(!zeroPreservingArgs.empty());
      if (std::includes(zeroArgs.begin(), zeroArgs.end(),
                        zeroPreservingArgs.begin(), zeroPreservingArgs.end())) {
        expr = IndexExpr();
        return;
      }
    }
    if (rewritten) {
      expr = new CallIntrinsicNode(op->func, args);
    }
    else {
      expr = op;
    }
  }

  void visit(const ReductionNode* op) {
    IndexExpr a = rewrite(op->a);
    if (!a.defined()) {
      expr = IndexExpr();
    }
    else if (a == op->a) {
      expr = op;
    }
    else {
      expr = new ReductionNode(op->op, op->var, a);
    }
  }


  void visit(const AssignmentNode* op) {
    IndexExpr rhs = rewrite(op->rhs);
    if (!rhs.defined()) {
      stmt = IndexStmt();
      zeroedVars.insert(op->lhs.getTensorVar());
    }
    else if (rhs == op->rhs) {
      stmt = op;
    }
    else {
      stmt = new AssignmentNode(op->lhs, rhs, op->op);
    }
  }

  void visit(const YieldNode* op) {
    IndexExpr expr = rewrite(op->expr);
    if (expr == op->expr) {
      stmt = op;
    }
    else {
      stmt = new YieldNode(op->indexVars, expr);
    }
  }

  void visit(const ForallNode* op) {
    IndexStmt body = rewrite(op->stmt);
    if (!body.defined()) {
      stmt = IndexStmt();
    }
    else if (body == op->stmt) {
      stmt = op;
    }
    else {
      stmt = new ForallNode(op->indexVar, body, op->merge_strategy, op->parallel_unit, op->output_race_strategy, op->unrollFactor);
    }
  }

  void visit(const ForallManyNode* op) {
    bool rewritten = false;
    std::vector<IndexStmt> newStmts;
    for (const auto &stmt: op->stmts){
      IndexStmt s = rewrite(stmt);
      if (s != stmt) {
        rewritten = true;
      }
      newStmts.push_back(s);
    }

    if (rewritten){
      stmt = op;
    }
    else {
        stmt = new ForallManyNode(op->indexVar, newStmts);
    }
  }

  void visit(const WhereNode* op) {
    IndexStmt producer = rewrite(op->producer);
    IndexStmt consumer = rewrite(op->consumer);
    if (!consumer.defined()) {
      stmt = IndexStmt();
    }
    else if (!producer.defined()) {
      stmt = consumer;
    }
    else if (producer == op->producer && consumer == op->consumer) {
      stmt = op;
    }
    else {
      stmt = new WhereNode(consumer, producer);
    }
  }

  void visit(const DimReductionNode* op) {
    IndexStmt producer = rewrite(op->producer);
    IndexStmt consumer = rewrite(op->consumer);
    if (!consumer.defined()) {
      stmt = IndexStmt();
    }
    else if (!producer.defined()) {
      stmt = consumer;
    }
    else if (producer == op->producer && consumer == op->consumer) {
      stmt = op;
    }
    else {
      stmt = new DimReductionNode(consumer, producer, op->temps);
    }
  }

  void visit(const AccelerateNode* op) {
    IndexStmt producer = rewrite(op->producer);
    IndexStmt consumer = rewrite(op->consumer);
    if (!consumer.defined()) {
      stmt = IndexStmt();
    }
    else if (!producer.defined()) {
      stmt = consumer;
    }
    else if (producer == op->producer && consumer == op->consumer) {
      stmt = op;
    }
    else {
      stmt = new AccelerateNode(consumer, producer, op->accelGen);
    }
  }

  void visit(const InterfaceCallNode* op) {
    IndexStmt producer = rewrite(op->producer);

    if (!producer.defined()) {
      stmt = op;
    }
    else if (producer == op->producer) {
      stmt = op;
    }
    else {
      stmt = new InterfaceCallNode(op->producer, op->codeGen, op->temp);
    }
  }

  void visit(const SequenceNode* op) {
    taco_not_supported_yet;
  }

  void visit(const AssembleNode* op) {
    taco_not_supported_yet;
  }

  void visit(const MultiNode* op) {
    taco_not_supported_yet;
  }

  void visit(const SuchThatNode* op) {
    IndexStmt body = rewrite(op->stmt);
    if (!body.defined()) {
      stmt = IndexStmt();
    }
    else if (body == op->stmt) {
      stmt = op;
    }
    else {
      stmt = new SuchThatNode(body, op->predicate);
    }
  }
};

IndexExpr zero(IndexExpr expr, const set<Access>& zeroed) {
  return Zero(zeroed).rewrite(expr);
}

IndexStmt zero(IndexStmt stmt, const std::set<Access>& zeroed) {
  return Zero(zeroed).rewrite(stmt);
}

// Attempts to infer the fill value of a given expression. If we cannot infer the value, an empty expression
// is returned
struct fillValueInferrer : IndexExprRewriterStrict {
  public:
    virtual void visit(const AccessNode* op) {
      expr = op->tensorVar.getFill();
    };

    virtual void visit(const LiteralNode* op) {
      expr = op;
    }

    virtual void visit(const NegNode* op) {
      IndexExpr a = rewrite(op->a);
      if(equals(a, Literal::zero(a.getDataType()))) {
        expr = a;
        return;
      }
      expr = IndexExpr();
    }

    virtual void visit(const AddNode* op) {
      IndexExpr a = rewrite(op->a);
      IndexExpr b = rewrite(op->b);

      if(equals(a, Literal::zero(a.getDataType())) && isa<Literal>(b)) {
        expr = b;
        return;
      }

      if(equals(b, Literal::zero(b.getDataType())) && isa<Literal>(a)) {
        expr = a;
        return;
      }

      expr = IndexExpr();
    }

    virtual void visit(const SubNode* op) {
      IndexExpr a = rewrite(op->a);
      IndexExpr b = rewrite(op->b);

      if(equals(b, Literal::zero(b.getDataType())) && isa<Literal>(a)) {
        expr = a;
        return;
      }

      expr = IndexExpr();
    }

    virtual void visit(const MulNode* op) {
      IndexExpr a = rewrite(op->a);
      IndexExpr b = rewrite(op->b);

      if(equals(a, Literal::zero(a.getDataType()))) {
        expr = a;
        return;
      }

      if(equals(b, Literal::zero(b.getDataType()))) {
        expr = b;
        return;
      }

      expr = IndexExpr();
    }

    virtual void visit(const DivNode* op) {
      IndexExpr a = rewrite(op->a);
      IndexExpr b = rewrite(op->b);

      if(equals(a, Literal::zero(a.getDataType()))) {
        expr = a;
        return;
      }

      expr = IndexExpr();
    }

    virtual void visit(const SqrtNode* op) {
      IndexExpr a = rewrite(op->a);
      if(equals(a, Literal::zero(a.getDataType()))) {
        expr = a;
        return;
      }
      expr = IndexExpr();
    }

    virtual void visit(const CastNode* op) {
      expr = IndexExpr();
    }

    virtual void visit(const CallNode* op) {
      Annihilator annihilator = findProperty<Annihilator>(op->properties);
      if(annihilator.defined()) {
        IndexExpr e = annihilator.annihilates(op->args);
        if(e.defined()) {
          expr = e;
          return;
        }
      }

      Identity identity = findProperty<Identity>(op->properties);
      if(identity.defined()) {
        IndexExpr e = identity.simplify(op->args);
        if(e.defined()) {
          expr = e;
          return;
        }
      }

      expr = IndexExpr();
    }

    virtual void visit(const CallIntrinsicNode*) {
      // TODO Implement or remove this
      taco_not_supported_yet;
    }

    virtual void visit(const ReductionNode*) {
      expr = IndexExpr();
    }

    virtual void visit(const IndexVarNode*) {
      expr = IndexExpr();
    }
  };


IndexExpr inferFill(IndexExpr expr) {
  return fillValueInferrer().rewrite(expr);
}

bool hasNoForAlls(IndexStmt stmt) {

  bool noForAlls = true;
  match(stmt,
        std::function<void(const ForallNode*)>([&](const ForallNode* op) {
          noForAlls = false;
        })
  );
  return noForAlls;
}

IndexStmt generatePackStmt(TensorVar tensor, 
                           std::string otherName, Format otherFormat, 
                           std::vector<IndexVar> indexVars, 
                           bool otherIsOnRight) { 

  const Type type = tensor.getType();
  TensorVar other(otherName, type, otherFormat);

  const Format format = tensor.getFormat();
  IndexStmt packStmt = otherIsOnRight ? 
                       (tensor(indexVars) = other(indexVars)) : 
                       (other(indexVars) = tensor(indexVars));

  for (int i = format.getOrder() - 1; i >= 0; --i) {
    int mode = format.getModeOrdering()[i];
    packStmt = forall(indexVars[mode], packStmt);
  }

  bool doAppend = true;
  const Format lhsFormat = otherIsOnRight ? format : otherFormat;
  for (int i = lhsFormat.getOrder() - 1; i >= 0; --i) {
    const auto modeFormat = lhsFormat.getModeFormats()[i];
    if (modeFormat.isBranchless() && i != 0) {
      const auto parentModeFormat = lhsFormat.getModeFormats()[i - 1];
      if (parentModeFormat.isUnique() || !parentModeFormat.hasAppend()) {
        doAppend = false;
        break;
      }
    }
  }
  if (!doAppend) {
    packStmt = packStmt.assemble(otherIsOnRight ? tensor : other, AssembleStrategy::Insert);
  }

  return packStmt; 
}

IndexStmt generatePackCOOStmt(TensorVar tensor, 
                              std::vector<IndexVar> indexVars, bool otherIsOnRight) {

  const std::string tensorName = tensor.getName();
  const Format format = tensor.getFormat();

  const Format bufferFormat = COO(format.getOrder(), false, true, false, 
                                  format.getModeOrdering());

  return generatePackStmt(tensor, tensorName + "_COO", bufferFormat, indexVars, otherIsOnRight);
}
}
