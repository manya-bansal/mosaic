#ifndef TEST_INTERFACE_H
#define TEST_INTERFACE_H

#include "taco/index_notation/index_notation.h"
#include "taco/type.h"
#include  "taco/index_notation/accel_interface.h"

using namespace taco;

class Test1 : public AbstractFunctionInterface{
    public: 
        Test1() : x(TensorVar(Type(taco::Float32, {Dimension()}), dense)),
                  y(TensorVar(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()) {};

        IndexExpr getRHS() const override {return x(i);}
        IndexExpr getLHS() const override {return x(i);}
        std::vector<Argument> getArguments() const override {return {new TensorVarArg(x)};}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "test";}

    private: 
        TensorVar x;
        TensorVar y;
        IndexVar i;
};


#endif 