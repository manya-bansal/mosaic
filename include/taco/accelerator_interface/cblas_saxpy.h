#ifndef CBLAS_INTERFACE_H
#define CBLAS_INTERFACE_H

#include "taco/index_notation/index_notation.h"
#include "taco/type.h"
#include "taco/accelerator_notation/accel_interface.h"

using namespace taco;

class Saxpy : public AbstractFunctionInterface{
    public: 
        Saxpy() : x(TensorVar(Type(taco::Float32, {Dimension()}), dense)),
                  y(TensorVar(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()) {};

        IndexExpr getRHS() const override {return x(i) + y(i);}
        IndexExpr getLHS() const override {return x(i);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new DimArg(i), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new TensorVarArg(y), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new TensorVarArg(x), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1)
                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override{return "cblas_saxpy";}
        bool checkerFunction(IndexStmt stmt) const override{return true;}

    private: 
        TensorVar x;
        TensorVar y;
        IndexVar i;
};

class Sdsdot : public AbstractFunctionInterface{
    public: 
        Sdsdot() : x(TensorVar(Type(taco::Float32, {Dimension()}), dense)),
                  y(TensorVar(Type(taco::Float32, {Dimension()}), dense)),
                  s(TensorVar(Type(taco::Float32))), i(IndexVar()) {};

        IndexExpr getRHS() const override {return x(i) * y(i);}
        IndexExpr getLHS() const override {return s;}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new DimArg(i), 
                                                    new LiteralArg(Datatype(taco::UInt32), 0),
                                                    new TensorVarArg(y), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new TensorVarArg(x), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1)
                                                };}
        std::string getReturnType() const override {return "float";}
        std::string getFunctionName() const override{return "cblas_sdsdot";}

    private: 
        TensorVar x;
        TensorVar y;
        TensorVar s;
        IndexVar i;
};



#endif 