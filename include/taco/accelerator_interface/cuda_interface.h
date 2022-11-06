#ifndef CUDA_INTERFACE_H
#define CUDA_INTERFACE_H

#include "taco/index_notation/index_notation.h"
#include "taco/type.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/accelerator_notation/accelerator_notation.h"

using namespace taco; 

// works for sqaure!!!
class CudaSpmv : public AbstractFunctionInterface{
    public: 
        CudaSpmv() : x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), Format{Dense, Sparse})), 
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  s(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()),
                  j(IndexVar()) {};
        AcceleratorStmt getStmt() const override {return y(i) = x(i, j)*s(j);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new TensorName(x),
                                                    new TensorObjectArg(s),
                                                    new TensorObjectArg(y),
                                                    new DimArg(i),
                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "cuda_spmv";}
    private: 
        TensorObject x;
        TensorObject y;
        TensorObject s;
        IndexVar i;
        IndexVar j;
};

#endif