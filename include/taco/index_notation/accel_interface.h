#ifndef ACCEL_INTERFACE_H
#define ACCEL_INTERFACE_H

#include "taco/index_notation/index_notation.h"
#include "taco/lower/iterator.h"

namespace taco {

class TransferWithArgs{
  public:
    TransferWithArgs() = default;

    TransferWithArgs(const std::string& name, const std::string& returnType , const std::vector<ir::Expr>& args) : name(name), returnType(returnType), args(args) {};
  
    std::string name;
    std::string returnType;
    std::vector<ir::Expr> args; 

};

std::ostream& operator<<(std::ostream&, const TransferWithArgs&);

class TransferLoad{
  public:
    TransferLoad() = default;
    TransferLoad(const std::string& name, const std::string& returnType) : name(name), returnType(returnType) {};

    template <typename... Exprs>
    TransferWithArgs operator()(const Exprs&... exprs){
      return TransferWithArgs(name, returnType, {exprs...});
    }

  private:
    std::string name;
    std::string returnType; 
};

class TransferStore{
  public:
    TransferStore() = default;
    TransferStore(const std::string& name, const std::string& returnType) : name(name), returnType(returnType){};

    template <typename... Exprs>
    TransferWithArgs operator()(const Exprs&... exprs){
      return TransferWithArgs(name, returnType, {exprs...});
    }

  private:
    std::string name;
    std::string returnType; 
};


// QUESTION: do we need different functions for runtime versus compile time 
// conversions (are there good cases for compile time conversions?)

class TransferType{
  public:
    TransferType(std::string name, taco::TransferLoad transferLoad, taco::TransferStore transferStore);
    // TransferType(TensorVar tensorVar);
  private:
    struct Content;
    std::shared_ptr<Content> content;

};

class AccelerateCodeGenerator {
  public: 
      AccelerateCodeGenerator(taco::IndexExpr expr, std::string functionName, std::vector<ir::Expr> args, std::function<bool(IndexExpr)> checker) :
                          expr(expr), functionName(functionName), args(args), checker(checker) {};

      AccelerateCodeGenerator() = default;

      taco::IndexExpr getExpr() { return expr; };

      taco::IndexExpr expr;
      std::string functionName;
      std::vector<ir::Expr> args;
      std::function<bool(IndexExpr)> checker;

};



// class packData{
//   packData()
// }

// class ReformatData{

//   ReformatData(std::string name, taco::TransferLoad transferLoad, taco::TransferStore transferStore);

//   private:
//     struct Content;
//     std::shared_ptr<Content> content;
// };



}
#endif