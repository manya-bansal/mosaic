#ifndef ACCEL_INTERFACE_H
#define ACCEL_INTERFACE_H

#include "taco/index_notation/index_notation.h"
#include "taco/lower/iterator.h"
#include "taco/type.h"

// #include "taco/index_notation/internal_args.h"


namespace taco {

struct TransferTypeArgs;
// struct TensorPropertiesArgs;
class TensorVar;
template <typename CType>
class Tensor;

enum ArgType {DIM, TENSORVAR, TENSOR, EXPR, LITERAL, USER_DEFINED, UNKNOWN};

// QUESTION: do we need different functions for runtime versus compile time 
// conversions (are there good cases for compile time conversions?)

class Argument : public util::IntrusivePtr<const TransferTypeArgs> {
  public: 
    Argument() : IntrusivePtr(nullptr) {}
    Argument(TransferTypeArgs * arg) : IntrusivePtr(arg) {}

    template<typename T>
    const T* getNode() const {
      return static_cast<const T*>(ptr);
    }

    const TransferTypeArgs* getNode() const {
      return ptr;
    }

    ArgType getArgType() const;

    
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

struct TensorVarArg : public TransferTypeArgs{
    explicit TensorVarArg(const TensorVar& t) : TransferTypeArgs(TENSOR), t(t) {}

    std::ostream& print(std::ostream& os) const override;
    TensorVar t; 
};

struct irExprArg : public TransferTypeArgs{
    explicit irExprArg(const ir::Expr& irExpr) : TransferTypeArgs(EXPR), irExpr(irExpr) {}

    std::ostream& print(std::ostream& os) const override;

    ir::Expr irExpr; 
};

class Dim{
  public:
    explicit Dim(const IndexVar& indexVar) : indexVar(indexVar) {}
    IndexVar indexVar;
};

struct DimArg : public TransferTypeArgs{
    explicit DimArg(const Dim& dim): TransferTypeArgs(DIM), indexVar(dim.indexVar) {}

    std::ostream& print(std::ostream& os) const override;

    IndexVar indexVar;
};

struct TensorArg : public TransferTypeArgs{
    template <typename CType>
    explicit TensorArg(const Tensor<CType>& tensor) : 
    TransferTypeArgs(TENSOR), irExpr(ir::Var::make(tensor.getName(), tensor.getComponentType(),true, true)) {}

    std::ostream& print(std::ostream& os) const override;

    ir::Expr irExpr; 
};


struct LiteralArg : public TransferTypeArgs{
    template <typename T> 
    LiteralArg(Datatype datatype, T val) 
      : TransferTypeArgs(LITERAL), datatype(datatype) {
        this->val = malloc(sizeof(T));
        *static_cast<T*>(this->val) = val;
    }

    ~LiteralArg() {
      free(val);
    }

    template <typename T> T getVal() const {
      return *static_cast<T*>(val);
    }

    std::ostream& print(std::ostream& os) const override;

    void * val; 
    Datatype datatype;
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

};

std::ostream& operator<<(std::ostream&, const TransferWithArgs&);

inline void addArg(std::vector<Argument>& argument, const TensorVar& t) { argument.push_back(new TensorVarArg(t)); };
inline void addArg(std::vector<Argument>& argument, const Argument&  arg) { argument.push_back(arg); };
inline void addArg(std::vector<Argument>& argument, TransferWithArgs * arg) { argument.push_back(arg); };
inline void addArg(std::vector<Argument>& argument, const Dim& dim) { argument.push_back(new DimArg(dim)); };
inline void addArg(std::vector<Argument>& argument, const int32_t& integer) { argument.push_back(new LiteralArg(Datatype(UInt32), integer)); };

template <typename CType>
inline void addArg(std::vector<Argument>& argument, const Tensor<CType>& t) { argument.push_back(new TensorArg(t)); };

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
      return new TransferWithArgs(name, returnType, argument);
    }

    template <typename FirstT, typename ...Args>
    Argument operator()(FirstT first, Args...remaining){
      std::vector<Argument> argument;
      addArg(argument, first, remaining...);
      return  new TransferWithArgs(name, returnType, argument); 
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
      return new TransferWithArgs(name, returnType, argument);
    }

    template <typename FirstT, typename ...Args>
    Argument operator()(FirstT first, Args...remaining){
      std::vector<Argument> argument;
      addArg(argument, first, remaining...);
      return  new TransferWithArgs(name, returnType, argument);
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



class ForeignFunctionDescription {
  public: 
    ForeignFunctionDescription() = default;

    ForeignFunctionDescription( const std::string& functionName, const std::string& returnType, const taco::IndexExpr& lhs, const taco::IndexExpr& rhs,  const std::vector<Argument>& args, 
                                const std::vector<TensorVar>& temporaries, std::function<bool(taco::IndexStmt)> checker) 
                                : functionName(functionName), returnType(returnType), lhs(lhs), rhs(rhs), args(args), temporaries(temporaries), checker(checker) {};
    
    ForeignFunctionDescription( const std::string& functionName, const std::string& returnType, const taco::IndexExpr& lhs, const taco::IndexExpr& rhs,  const std::vector<Argument>& args, 
                                const std::vector<TensorVar>& temporaries, std::function<bool(taco::IndexStmt)> checker,  const  std::map<TensorVar, std::set<std::string>>& propertites)
                                : functionName(functionName), returnType(returnType), lhs(lhs), rhs(rhs), args(args), temporaries(temporaries), checker(checker), propertites(propertites) {};

    ForeignFunctionDescription( const std::string& functionName, const std::string& returnType, const taco::IndexExpr& lhs, const taco::IndexExpr& rhs,
                                const std::vector<TensorVar>& temporaries, std::function<bool(taco::IndexStmt)> checker) 
                                : functionName(functionName), returnType(returnType), lhs(lhs), rhs(rhs), temporaries(temporaries), checker(checker) {};

    ForeignFunctionDescription( const std::string& functionName, const std::string& returnType, const taco::IndexExpr& lhs, const taco::IndexExpr& rhs,
                                const std::vector<TensorVar>& temporaries, std::function<bool(taco::IndexStmt)> checker,
                                const  std::map<TensorVar, std::set<std::string>>& propertites)
                                : functionName(functionName), returnType(returnType), lhs(lhs), rhs(rhs), temporaries(temporaries), checker(checker), propertites(propertites) {};

    template <typename Exprs> 
    ForeignFunctionDescription operator()(Exprs expr)
    {  std::vector<Argument> argument;
      addArg(argument, expr);
      return ForeignFunctionDescription(functionName, returnType, lhs, rhs, argument, temporaries, checker);
    }

    template <typename FirstT, typename ...Args>
    ForeignFunctionDescription operator()(FirstT first, Args...remaining){
        std::vector<Argument> argument;
        addArg(argument, first, remaining...);
        return ForeignFunctionDescription(functionName, returnType, lhs, rhs, argument, temporaries, checker);
    }

    std::string functionName;
    std::string returnType;
    taco::IndexExpr lhs;
    taco::IndexExpr rhs;
    std::vector<Argument> args;
    std::vector<taco::TensorVar> temporaries;
    std::function<bool(taco::IndexStmt)> checker;
    std::map<TensorVar, std::set<std::string>> propertites;

};

std::ostream& operator<<(std::ostream&, const ForeignFunctionDescription&);


class AcceleratorDescription {
  public:
    AcceleratorDescription(const TransferType& kernelTransfer, const std::vector<ForeignFunctionDescription>& funcDescriptions) : kernelTransfer(kernelTransfer), funcDescriptions(funcDescriptions) {};
    AcceleratorDescription(const TransferType& kernelTransfer, const std::vector<ForeignFunctionDescription>& funcDescriptions, const std::string& includeFile)
                           : kernelTransfer(kernelTransfer), funcDescriptions(funcDescriptions), includeFile(includeFile) {};

    TransferType kernelTransfer;
    std::vector<ForeignFunctionDescription> funcDescriptions;
    //files to include
    std::string includeFile;

};

class ConcreteAccelerateCodeGenerator {
  public: 
    // TODO: Add some error checking here so users 
    // dont pass in any legal IndexExpr
    ConcreteAccelerateCodeGenerator() = default;

    ConcreteAccelerateCodeGenerator(const std::string& functionName, const std::string& returnType, const taco::IndexExpr& lhs, const taco::IndexExpr& rhs, const std::vector<Argument>& args, 
                                    const std::vector<taco::TensorVar>& declarations)
                                    : functionName(functionName), returnType(returnType), lhs(lhs), rhs(rhs), args(args), declarations(declarations) {}

    ConcreteAccelerateCodeGenerator(const std::string& functionName, const std::string& returnType, taco::IndexExpr lhs, taco::IndexExpr rhs,
                                    const std::vector<taco::TensorVar>& declarations)
                                    : functionName(functionName), returnType(returnType), lhs(lhs), rhs(rhs), declarations(declarations) {}
  
    taco::IndexExpr getExpr() const     {return rhs;};
    taco::IndexExpr getLHS() const      {return lhs;};
    taco::IndexExpr getRHS() const      {return rhs;};
    std::vector<Argument> getArguments() const {return args;};
    std::string getReturnType() const   {return returnType;};
    std::string getFunctionName() const {return functionName;};

    template <typename Exprs> 
    ConcreteAccelerateCodeGenerator operator()(Exprs expr)
    {  std::vector<Argument> argument;
      addArg(argument, expr);
      return ConcreteAccelerateCodeGenerator(functionName, returnType, lhs, rhs, argument, declarations);
    }

    template <typename FirstT, typename ...Args>
    ConcreteAccelerateCodeGenerator operator()(FirstT first, Args...remaining){
        std::vector<Argument> argument;
        addArg(argument, first, remaining...);
        return ConcreteAccelerateCodeGenerator(functionName, returnType, lhs, rhs, argument, declarations);
    }

      std::string functionName;
      std::string returnType;
      //there is a smal problem with using 
      //indexStmts
      taco::IndexExpr lhs;
      taco::IndexExpr rhs;
      std::vector<Argument> args;
      std::vector<taco::TensorVar> declarations;

};

std::ostream& operator<<(std::ostream&,  const ConcreteAccelerateCodeGenerator&);

}
#endif