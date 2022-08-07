#ifndef CBLAS_INTERFACE_H
#define CBLAS_INTERFACE_H

#include "taco/index_notation/index_notation.h"
#include "taco/type.h"
#include  "taco/index_notation/accel_interface.h"

using namespace taco;

class Saxpy : public taco::AbstractFunctionInterface{
    public: 
        Saxpy() : x(taco::TensorVar(taco::Type(taco::Float32, {taco::Dimension()}), taco::dense)),
                  y(taco::TensorVar(taco::Type(taco::Float32, {taco::Dimension()}), taco::dense)),
                  i(taco::IndexVar()) {};

        taco::IndexExpr getRHS() const {return x(i) + y(i);}
        taco::IndexExpr getExpr() const {return x(i) + y(i);}
        taco::IndexExpr getLHS() const {return x(i);}
        std::vector<Argument> getArguments() const {return {new DimArg(i), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new TensorVarArg(y), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new TensorVarArg(x), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1)};}
        std::string getReturnType() const {return "void";}
        std::string getFunctionName() const {return "cblas_saxpy";}

    private: 
        taco::TensorVar x;
        taco::TensorVar y;
        taco::IndexVar i;
};


#endif 