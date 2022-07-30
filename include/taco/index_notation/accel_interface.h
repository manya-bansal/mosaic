#ifndef ACCEL_INTERFACE_H
#define ACCEL_INTERFACE_H

#include <variant>
#include "taco/index_notation/index_notation.h"
#include "taco/lower/iterator.h"
#include "taco/tensor.h"


namespace taco {

struct TransferTypeArgs;
struct TensorPropertiesArgs;

enum ArgType {USER_DEFINED, INTERNAL, UNKNOWN};

class Argument : public util::IntrusivePtr<const TransferTypeArgs> {
  public: 
    Argument() : IntrusivePtr(nullptr) {}
    Argument(TransferTypeArgs * arg) : IntrusivePtr(arg) {}

    ArgType getArgType() const;

    template<typename T>
    const T* getNode() const {
      return static_cast<const T*>(ptr);
    }

    const TransferTypeArgs* getNode() const {
    return ptr;
    }
};

//We need to types of args: args that we provide to the user
//dimension of a user etc, and a way for them to call a special function

struct TransferTypeArgs : public util::Manageable<TransferTypeArgs>{
    TransferTypeArgs() : argType(UNKNOWN) {}
    TransferTypeArgs(ArgType argType) : argType(argType) {}
    virtual ~TransferTypeArgs() = default;
    virtual void lower() const {  };

    virtual std::ostream& print(std::ostream& os) const {
        os << "Printing a TransferTypeArg" << std::endl;
        return os;
    };

    ArgType argType;

};

std::ostream& operator<<(std::ostream&,  const Argument&);

struct TensorPropertiesArgs : public TransferTypeArgs{

    TensorPropertiesArgs(ir::Expr irExpr) : TransferTypeArgs(INTERNAL), irExpr(irExpr) {};

    TensorPropertiesArgs(TensorVar t);

    // template <typename T>
    // TensorPropertiesArgs(T t){
    //   irExpr = ir::Var::make(t.getName(), t.getComponentType(),true, true);
    // }

    void lower() const;

    friend std::ostream& operator<<(std::ostream&, const TensorPropertiesArgs&);

    std::ostream& print(std::ostream& os) const;

    ir::Expr irExpr; 
    ArgType argType;

};



struct TransferWithArgs : public TransferTypeArgs{
    TransferWithArgs() = default;

    TransferWithArgs(const std::string& name, const std::string& returnType , const std::vector<Argument>& args) : TransferTypeArgs(USER_DEFINED), name(name), returnType(returnType), args(args) {};

    // TransferWithArgs(const std::string& name, const std::string& returnType, const TransferWithArgs& transferWithArgs)

    std::string getName() const { return name; };
    std::string getReturnType() const { return returnType; };
    std::vector<Argument> getArgs() const { return args; };

    void lower() const;

    std::ostream& print(std::ostream& os) const;
    
    std::string name;
    std::string returnType;
    std::vector<Argument> args; 
    ArgType argType;

};

std::ostream& operator<<(std::ostream&, const TransferWithArgs&);

inline void addArg(std::vector<Argument>& argument, TensorVar t) { argument.push_back(new TensorPropertiesArgs(t)); };
inline void addArg(std::vector<Argument>& argument, Argument arg) { argument.push_back(arg); };



template <typename T, typename ...Next>
void addArg(std::vector<Argument>& argument, T first, Next...next){
    addArg(argument, first);
    addArg(argument, (next)...); 
}


class TransferLoad{
  public:
    TransferLoad() = default;
    TransferLoad(const std::string& name, const std::string& returnType) : name(name), returnType(returnType) {};

    template <typename Exprs> 
    Argument operator()(Exprs expr)
    { 
      std::vector<Argument> argument;
      addArg(argument, expr);
      return new TransferWithArgs(name, returnType, {argument});
    }

    template <typename FirstT, typename ...Args>
    Argument operator()(FirstT first, Args...remaining){
      std::vector<Argument> argument;
      addArg(argument, first, remaining...);
      return  new TransferWithArgs(name, returnType, {argument});
    }

  private:
    std::string name;
    std::string returnType; 
};

class TransferStore{
  public:
    TransferStore() = default;
    TransferStore(const std::string& name, const std::string& returnType) : name(name), returnType(returnType){};
    

    template <typename Exprs> 
    Argument operator()(Exprs expr)
    { 
      std::vector<Argument> argument;
      addArg(argument, expr);
      return new TransferWithArgs(name, returnType, {argument});
    }

    template <typename FirstT, typename ...Args>
    Argument operator()(FirstT first, Args...remaining){
      std::vector<Argument> argument;
      addArg(argument, first, remaining...);
      return  new TransferWithArgs(name, returnType, {argument});
    }

  private:
    std::string name;
    std::string returnType; 
};

class TransferType{
  public:
    TransferType(std::string name, taco::TransferLoad transferLoad, taco::TransferStore transferStore);
  private:
    struct Content;
    std::shared_ptr<Content> content;

};


// QUESTION: do we need different functions for runtime versus compile time 
// conversions (are there good cases for compile time conversions?)




class ConcreteAccelerateCodeGenerator {
  public: 

      ConcreteAccelerateCodeGenerator() = default;

      ConcreteAccelerateCodeGenerator(taco::IndexExpr expr, const std::string& functionName, const std::vector<ir::Expr>& args, std::function<bool(IndexExpr)> checker) :
                          expr(expr), functionName(functionName), args(args), checker(checker) {};

    
      taco::IndexExpr getExpr() { return expr; };

      taco::IndexExpr expr;
      std::string functionName;
      std::vector<ir::Expr> args;
      std::function<bool(taco::IndexExpr)> checker;

};


class ForeignFunctionDescription {
  public: 
    ForeignFunctionDescription() = default;
    ForeignFunctionDescription( const IndexStmt& targetStmt, const std::string& functionName, const std::string& returnType, const std::vector<Argument>& args, 
                                const std::vector<TensorVar>& temporaries, std::function<bool(taco::IndexStmt)> checker) 
                                : targetStmt(targetStmt), functionName(functionName), returnType(returnType), args(args), temporaries(temporaries), checker(checker) {};

    ForeignFunctionDescription( const IndexStmt& targetStmt, const std::string& functionName, const std::string& returnType,
                                    const std::vector<TensorVar>& temporaries, std::function<bool(taco::IndexStmt)> checker) 
                                    : targetStmt(targetStmt), functionName(functionName), returnType(returnType), temporaries(temporaries), checker(checker) {};

    template <typename... Exprs> 
    ForeignFunctionDescription operator()(const Exprs... expr){
        return ForeignFunctionDescription(targetStmt, functionName, returnType, {expr...}, temporaries, checker);
    }

    taco::IndexStmt targetStmt;
    std::string functionName;
    std::string returnType;
    std::vector<Argument> args;
    std::vector<taco::TensorVar> temporaries;
    std::function<bool(taco::IndexStmt)> checker;

};


class AcceleratorDescription {
  public:
    AcceleratorDescription(TransferType kernelTransfer, std::vector<ForeignFunctionDescription> funcDescriptions) : kernelTransfer(kernelTransfer), funcDescriptions(funcDescriptions) {};

    TransferType kernelTransfer;
    std::vector<ForeignFunctionDescription> funcDescriptions;
    //files to include

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