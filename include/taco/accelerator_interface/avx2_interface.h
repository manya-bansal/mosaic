#ifndef AVX2_INTERFACE_H
#define AVX2_INTERFACE_H

#include "taco/index_notation/index_notation.h"
#include "taco/type.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/accelerator_notation/accelerator_notation.h"

class AVXSaxpy : public AbstractFunctionInterface{
    public: 
        AVXSaxpy() :x(TensorObject(Type(taco::Float32, {8}), dense)),
                    y(TensorObject(Type(taco::Float32, {8}), dense)),
                    z(TensorObject(Type(taco::Float32, {8}), dense)),
                    i(IndexVar()),
                    var(DeclVar("__m256", "var1")),
                    var2(DeclVar("__m256", "var2")),
                    result(DeclVar("__m256", "result")) {};

        AcceleratorStmt getStmt() const override {return z(i) = x(i) + y(i);}
        std::vector<Argument> getArguments() const override {
                                                return 
                                                {   new TensorObjectArg(z),
                                                    new DeclVarArg(result)
                                                }; }

        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "_mm256_storeu_ps";}
        std::vector<Argument>  callBefore() const override {
                                taco::TransferLoad _mm256_load_ps("_mm256_loadu_ps", "__m256");
                                taco::TransferLoad _mm256_add_ps("_mm256_add_ps", "__m256");
                                return { 
                                         var = _mm256_load_ps(x),
                                         var2 = _mm256_load_ps(y),
                                         result = _mm256_add_ps(var, var2)};
                            }

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject z;
        IndexVar i;
        DeclVar var;
        DeclVar var2;
        DeclVar result;

};

#endif 