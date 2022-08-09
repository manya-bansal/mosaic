#ifndef ACCELERATOR_NOTATION_NODES_H
#define ACCELERATOR_NOTATION_NODES_H

#include <vector>
#include <memory>
#include <functional>
#include <numeric>
#include <functional>

#include "taco/type.h"
#include "taco/util/collections.h"
#include "taco/util/comparable.h"
#include "taco/type.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/accelerator_notation/accelerator_notation_nodes_abstract.h"
#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/accelerator_notation/accelerator_notation_visitor.h"
#include "taco/index_notation/intrinsic.h"
#include "taco/util/strings.h"

namespace taco {

struct AcceleratorAccessNode : public AcceleratorExprNode {
  AcceleratorAccessNode(TensorObject tensorObject, const std::vector<IndexVar>& indices, bool isAccessingStructure)
      : AcceleratorExprNode(isAccessingStructure ? Bool : tensorObject.getType().getDataType()), 
        tensorObject(tensorObject), indexVars(indices), 
        isAccessingStructure(isAccessingStructure) {
  }

  void accept(AcceleratorExprVisitorStrict* v) const {
    v->visit(this);
  }

  virtual void setAssignment(const Assignment& assignment) {}

  TensorObject tensorObject;
  std::vector<IndexVar> indexVars;
  bool isAccessingStructure;

protected:
  /// Initialize an AccessNode with just a TensorVar. If this constructor is used,
  /// then indexVars must be set afterwards.
  explicit AcceleratorAccessNode(TensorObject tensorObject) : 
      AcceleratorExprNode((tensorObject.getType().getDataType())), 
      tensorObject(tensorObject), isAccessingStructure(false) {}
};

struct AcceleratorLiteralNode : public AcceleratorExprNode {
  template <typename T> 
  explicit AcceleratorLiteralNode(T val) : AcceleratorExprNode(type<T>()) {
    this->val = malloc(sizeof(T));
    *static_cast<T*>(this->val) = val;
  }

  ~AcceleratorLiteralNode() {
    free(val);
  }

  void accept(AcceleratorExprVisitorStrict* v) const {
    v->visit(this);
  }

  template <typename T> T getVal() const {
    taco_iassert(getDataType() == type<T>())
        << "Attempting to get data of wrong type";
    return *static_cast<T*>(val);
  }

  void* val;
};

// struct UnaryExprNode : public IndexExprNode {
//   IndexExpr a;

// protected:
//   UnaryExprNode(IndexExpr a) : IndexExprNode(a.getDataType()), a(a) {}
// };


// struct NegNode : public UnaryExprNode {
//   NegNode(IndexExpr operand) : UnaryExprNode(operand) {}

//   void accept(IndexExprVisitorStrict* v) const {
//     v->visit(this);
//   }
// };


// struct BinaryExprNode : public IndexExprNode {
//   virtual std::string getOperatorString() const = 0;

//   IndexExpr a;
//   IndexExpr b;

// protected:
//   BinaryExprNode() : IndexExprNode() {}
//   BinaryExprNode(IndexExpr a, IndexExpr b)
//       : IndexExprNode(max_type(a.getDataType(), b.getDataType())), a(a), b(b) {}
// };


// struct AddNode : public BinaryExprNode {
//   AddNode() : BinaryExprNode() {}
//   AddNode(IndexExpr a, IndexExpr b) : BinaryExprNode(a, b) {}

//   std::string getOperatorString() const {
//     return "+";
//   }

//   void accept(IndexExprVisitorStrict* v) const {
//     v->visit(this);
//   }
// };


// struct SubNode : public BinaryExprNode {
//   SubNode() : BinaryExprNode() {}
//   SubNode(IndexExpr a, IndexExpr b) : BinaryExprNode(a, b) {}

//   std::string getOperatorString() const {
//     return "-";
//   }

//   void accept(IndexExprVisitorStrict* v) const {
//     v->visit(this);
//   }
// };


// struct MulNode : public BinaryExprNode {
//   MulNode() : BinaryExprNode() {}
//   MulNode(IndexExpr a, IndexExpr b) : BinaryExprNode(a, b) {}

//   std::string getOperatorString() const {
//     return "*";
//   }

//   void accept(IndexExprVisitorStrict* v) const {
//     v->visit(this);
//   }
// };


// struct DivNode : public BinaryExprNode {
//   DivNode() : BinaryExprNode() {}
//   DivNode(IndexExpr a, IndexExpr b) : BinaryExprNode(a, b) {}

//   std::string getOperatorString() const {
//     return "/";
//   }

//   void accept(IndexExprVisitorStrict* v) const {
//     v->visit(this);
//   }
// };


// struct SqrtNode : public UnaryExprNode {
//   SqrtNode(IndexExpr operand) : UnaryExprNode(operand) {}

//   void accept(IndexExprVisitorStrict* v) const {
//     v->visit(this);
//   }

// };


// struct CastNode : public IndexExprNode {
//   CastNode(IndexExpr operand, Datatype newType);

//   void accept(IndexExprVisitorStrict* v) const {
//     v->visit(this);
//   }

//   IndexExpr a;
// };


// struct CallIntrinsicNode : public IndexExprNode {
//   CallIntrinsicNode(const std::shared_ptr<Intrinsic>& func,
//                     const std::vector<IndexExpr>& args); 

//   void accept(IndexExprVisitorStrict* v) const {
//     v->visit(this);
//   }

//   std::shared_ptr<Intrinsic> func;
//   std::vector<IndexExpr> args;
// };

// struct CallNode : public IndexExprNode {
//   typedef std::function<ir::Expr(const std::vector<ir::Expr>&)> OpImpl;
//   typedef std::function<IterationAlgebra(const std::vector<IndexExpr>&)> AlgebraImpl;

//   CallNode(std::string name, const std::vector<IndexExpr>& args, OpImpl lowerFunc,
//            const IterationAlgebra& iterAlg,
//            const std::vector<Property>& properties,
//            const std::map<std::vector<int>, OpImpl>& regionDefinitions,
//            const std::vector<int>& definedRegions);

//   CallNode(std::string name, const std::vector<IndexExpr>& args, OpImpl lowerFunc,
//            const IterationAlgebra& iterAlg,
//            const std::vector<Property>& properties,
//            const std::map<std::vector<int>, OpImpl>& regionDefinitions);

//   void accept(IndexExprVisitorStrict* v) const {
//     v->visit(this);
//   }

//   std::string name;
//   std::vector<IndexExpr> args;
//   OpImpl defaultLowerFunc;
//   IterationAlgebra iterAlg;
//   std::vector<Property> properties;
//   std::map<std::vector<int>, OpImpl> regionDefinitions;

//   // Needed to track which inputs have been exhausted so the lowerer can know which lower func to use
//   std::vector<int> definedRegions;

// private:
//   static Datatype inferReturnType(OpImpl f, const std::vector<IndexExpr>& inputs) {
//     std::function<ir::Expr(IndexExpr)> getExprs = [](IndexExpr arg) { return ir::Var::make("t", arg.getDataType()); };
//     std::vector<ir::Expr> exprs = util::map(inputs, getExprs);

//     if(exprs.empty()) {
//       return taco::Datatype();
//     }

//     return f(exprs).type();
//   }

//   static std::vector<int> definedIndices(std::vector<IndexExpr> args) {
//     std::vector<int> v;
//     for(int i = 0; i < (int) args.size(); ++i) {
//       if(args[i].defined()) {
//         v.push_back(i);
//       }
//     }
//     return v;
//   }
// };

// struct ReductionNode : public IndexExprNode {
//   ReductionNode(IndexExpr op, IndexVar var, IndexExpr a);

//   void accept(IndexExprVisitorStrict* v) const {
//      v->visit(this);
//   }

//   IndexExpr op;  // The binary reduction operator, which is a `BinaryExprNode`
//                  // with undefined operands)
//   IndexVar var;
//   IndexExpr a;
// };

// struct IndexVarNode : public IndexExprNode, public util::Comparable<IndexVarNode> {
//   IndexVarNode() = delete;
//   IndexVarNode(const std::string& name, const Datatype& type);

//   void accept(IndexExprVisitorStrict* v) const {
//     v->visit(this);
//   }

//   std::string getName() const;

//   friend bool operator==(const IndexVarNode& a, const IndexVarNode& b);
//   friend bool operator<(const IndexVarNode& a, const IndexVarNode& b);

// private:
//   struct Content;
//   std::shared_ptr<Content> content;
// };

// struct IndexVarNode::Content {
//   std::string name;
// };

// // Index Statements
// struct AssignmentNode : public IndexStmtNode {
//   AssignmentNode(const Access& lhs, const IndexExpr& rhs, const IndexExpr& op)
//       : lhs(lhs), rhs(rhs), op(op) {}

//   void accept(IndexStmtVisitorStrict* v) const {
//     v->visit(this);
//   }

//   Access    lhs;
//   IndexExpr rhs;
//   IndexExpr op;
// };

// struct YieldNode : public IndexStmtNode {
//   YieldNode(const std::vector<IndexVar>& indexVars, IndexExpr expr)
//       : indexVars(indexVars), expr(expr) {}

//   void accept(IndexStmtVisitorStrict* v) const {
//     v->visit(this);
//   }

//   std::vector<IndexVar> indexVars;
//   IndexExpr expr;
// };

// struct ForallNode : public IndexStmtNode {
//   ForallNode(IndexVar indexVar, IndexStmt stmt, MergeStrategy merge_strategy, ParallelUnit parallel_unit, OutputRaceStrategy  output_race_strategy, size_t unrollFactor = 0)
//       : indexVar(indexVar), stmt(stmt), merge_strategy(merge_strategy), parallel_unit(parallel_unit), output_race_strategy(output_race_strategy), unrollFactor(unrollFactor) {}

//   void accept(IndexStmtVisitorStrict* v) const {
//     v->visit(this);
//   }

//   IndexVar indexVar;
//   IndexStmt stmt;
//   MergeStrategy merge_strategy;
//   ParallelUnit parallel_unit;
//   OutputRaceStrategy  output_race_strategy;
//   size_t unrollFactor = 0;
// };

// struct WhereNode : public IndexStmtNode {
//   WhereNode(IndexStmt consumer, IndexStmt producer)
//       : consumer(consumer), producer(producer) {}

//   void accept(IndexStmtVisitorStrict* v) const {
//     v->visit(this);
//   }

//   IndexStmt consumer;
//   IndexStmt producer;
// };

// struct AccelerateNode : public IndexStmtNode {
//   AccelerateNode(IndexStmt consumer, IndexStmt producer, ConcreteAccelerateCodeGenerator accelGen)
//       : consumer(consumer), producer(producer), accelGen(accelGen) {}

//   void accept(IndexStmtVisitorStrict* v) const {
//     v->visit(this);
//   }

//   IndexStmt consumer;
//   IndexStmt producer;
//   ConcreteAccelerateCodeGenerator accelGen;
// };

// struct MultiNode : public IndexStmtNode {
//   MultiNode(IndexStmt stmt1, IndexStmt stmt2) : stmt1(stmt1), stmt2(stmt2) {}

//   void accept(IndexStmtVisitorStrict* v) const {
//     v->visit(this);
//   }

//   IndexStmt stmt1;
//   IndexStmt stmt2;
// };

// struct SuchThatNode : public IndexStmtNode {
//   SuchThatNode(IndexStmt stmt, std::vector<IndexVarRel> predicate) : stmt(stmt), predicate(predicate) {}

//   void accept(IndexStmtVisitorStrict* v) const {
//     v->visit(this);
//   }

//   IndexStmt stmt;
//   std::vector<IndexVarRel> predicate;
// };

// struct SequenceNode : public IndexStmtNode {
//   SequenceNode(IndexStmt definition, IndexStmt mutation)
//       : definition(definition), mutation(mutation) {}

//   void accept(IndexStmtVisitorStrict* v) const {
//     v->visit(this);
//   }

//   IndexStmt definition;
//   IndexStmt mutation;
// };

// struct AssembleNode : public IndexStmtNode {
//   AssembleNode(IndexStmt queries, IndexStmt compute, 
//                Assemble::AttrQueryResults results)
//       : queries(queries), compute(compute), results(results) {}

//   void accept(IndexStmtVisitorStrict* v) const {
//     v->visit(this);
//   }

//   IndexStmt queries;
//   IndexStmt compute;
//   Assemble::AttrQueryResults results;
// };


// /// Returns true if expression e is of type E.
// template <typename E>
// inline bool isa(const IndexExprNode* e) {
//   return e != nullptr && dynamic_cast<const E*>(e) != nullptr;
// }

// /// Casts the expression e to type E.
// template <typename E>
// inline const E* to(const IndexExprNode* e) {
//   taco_iassert(isa<E>(e)) <<
//       "Cannot convert " << typeid(e).name() << " to " << typeid(E).name();
//   return static_cast<const E*>(e);
// }

// /// Returns true if statement e is of type S.
// template <typename S>
// inline bool isa(const IndexStmtNode* s) {
//   return s != nullptr && dynamic_cast<const S*>(s) != nullptr;
// }

// /// Casts the index statement node s to subtype S.
// template <typename SubType>
// inline const SubType* to(const IndexStmtNode* s) {
//   taco_iassert(isa<SubType>(s)) <<
//       "Cannot convert " << typeid(s).name() << " to " << typeid(SubType).name();
//   return static_cast<const SubType*>(s);
// }

// template <typename I>
// inline const typename I::Node* getNode(const I& stmt) {
//   taco_iassert(isa<typename I::Node>(stmt.ptr));
//   return static_cast<const typename I::Node*>(stmt.ptr);
// }

}
#endif
