#ifndef TILE_INTERFACE_H
#define TILE_INTERFACE_H

#include "taco/index_notation/index_notation.h"
#include "taco/type.h"
#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/accelerator_notation/accel_interface.h"

using namespace taco;

class TileSaxpy : public AbstractFunctionInterface{
    public: 
        TileSaxpy() : x(TensorObject(Type(taco::Float32, {4}), dense)),
                  y(TensorObject(Type(taco::Float32, {4}), dense)),
                  i(IndexVar()) {};

        taco::AcceleratorStmt getStmt() const override{ return x(i) = x(i) + y(i);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new LiteralArg(Datatype(taco::UInt32), 4),
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new TensorObjectArg(y), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new TensorObjectArg(x), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1)
                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override{return "cblas_saxpy";}
        bool checkerFunction(IndexStmt stmt) const override{return true;}

    private: 
        TensorObject x;
        TensorObject y;
        IndexVar i;
};


#endif 