#ifndef TEST_INTERFACE_H
#define TEST_INTERFACE_H

#include "taco/index_notation/index_notation.h"
#include "taco/type.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/accelerator_notation/accelerator_notation.h"

using namespace taco;

class DotProduct : public AbstractFunctionInterface{
    public: 
        DotProduct() : x(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  s(TensorObject(Type(taco::Float32))),
                  i(IndexVar()) {};

        // IndexExpr getRHS() const override {return x(i);}
        // IndexExpr getLHS() const override {return x(i);}
        AcceleratorStmt getStmt() const override {return s = x(i) * y(i);}
        std::vector<Argument> getArguments() const override {return {new DimArg(i), new TensorObjectArg(x), new TensorObjectArg(y)};}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "dotProduct";}

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject s;
        IndexVar i;
};


#endif 