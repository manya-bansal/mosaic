#ifndef MKL_INTERFACE_H
#define MKL_INTERFACE_H

#include "taco/index_notation/index_notation.h"
#include "taco/type.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/accelerator_notation/accelerator_notation.h"

using namespace taco; 

// works for sqaure!!!
class MklSgemv : public AbstractFunctionInterface{
    public: 
        MklSgemv() : x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), Format{Dense, Dense})), 
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  s(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()),
                  j(IndexVar()),
                  a(DeclVar("float", "var1")),
                  b(DeclVar("MKL_INT", "var2")),
                  zero(DeclVar("float", "var4")),
                  dim(DeclVar("MKL_INT", "var3")) {};
        AcceleratorStmt getStmt() const override {return y(i) = x(i, j)*s(j);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new DimArg(i), 
                                                    new TensorObjectArg(x),
                                                    new TensorObjectArg(s),
                                                    new TensorObjectArg(y),
                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "sgemv_mkl_internal";}
    private: 
        TensorObject x;
        TensorObject y;
        TensorObject s;
        IndexVar i;
        IndexVar j;
        DeclVar a;
        DeclVar b;
        DeclVar dim;
        DeclVar zero;
};


class SparseMklSgemv : public AbstractFunctionInterface{
    public: 
        SparseMklSgemv() : x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), Format{Dense, Sparse})), 
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  s(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()),
                  j(IndexVar()),
                  a(DeclVar("float", "var1")),
                  b(DeclVar("MKL_INT", "var2")),
                  zero(DeclVar("float", "var4")),
                  dim(DeclVar("MKL_INT", "var3")) {};
        AcceleratorStmt getStmt() const override {return y(i) = x(i, j)*s(j);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new DimArg(i), 
                                                    new TensorName(x),
                                                    new TensorName(s),
                                                    new TensorName(y),
                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "mkl_scsrgemv_internal";}
    private: 
        TensorObject x;
        TensorObject y;
        TensorObject s;
        IndexVar i;
        IndexVar j;
        DeclVar a;
        DeclVar b;
        DeclVar zero;
        DeclVar dim;
};

//works on sqaure!!
class SparseMklMM : public AbstractFunctionInterface{
public:
    SparseMklMM() : 
                    x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}),  Format{Dense, Sparse})),
                    y(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}),  Format{Dense, Dense})),
                    z(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}),  Format{Dense, Dense})),
                    i(IndexVar()),
                    j(IndexVar()),
                    k(IndexVar()) {};

        AcceleratorStmt getStmt() const override {return z(i, k) = x(i, j) * y(j, k);}
        std::vector<Argument> getArguments() const override {
                                                return 
                                                {
                                                    new DimArg(i), 
                                                    new TensorName(x),
                                                    new TensorName(y),
                                                    new TensorName(z),
                                                };}

        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "mkl_sparse_s_mm_internal";}
    private: 
        TensorObject x;
        TensorObject y;
        TensorObject z;
        IndexVar i;
        IndexVar j;
        IndexVar k;
};


#endif 