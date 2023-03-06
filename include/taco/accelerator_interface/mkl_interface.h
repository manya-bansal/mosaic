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
                  j(IndexVar()) {};
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

class MklSymmgemv : public AbstractFunctionInterface{
    public: 
        MklSymmgemv() : x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), Format{Dense, Dense})), 
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  s(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()),
                  j(IndexVar()) {};
        AcceleratorStmt getStmt() const override {return y(i) = x(i, j)*s(j);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new DimArg(i), 
                                                    new TensorObjectArg(x),
                                                    new TensorObjectArg(s),
                                                    new TensorObjectArg(y),
                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "ssymv_mkl_internal";}
    private: 
        TensorObject x;
        TensorObject y;
        TensorObject s;
        IndexVar i;
        IndexVar j;
};

class MklMM : public AbstractFunctionInterface{
public:
    MklMM() : 
                    x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}),  Format{Dense, Dense})),
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
                                                    new TensorObjectArg(x),
                                                    new TensorObjectArg(y),
                                                    new TensorObjectArg(z),
                                                };}

        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "sgemm_mkl_internal";}
    private: 
        TensorObject x;
        TensorObject y;
        TensorObject z;
        IndexVar i;
        IndexVar j;
        IndexVar k;
};

class MklDot : public AbstractFunctionInterface{
    public: 
        MklDot() : x(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  s(TensorObject(Type(taco::Float32))),
                  i(IndexVar()) {};

        // IndexExpr getRHS() const override {return x(i);}
        // IndexExpr getLHS() const override {return x(i);}
        AcceleratorStmt getStmt() const override {return s = x(i) * y(i);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new DimArg(i), 
                                                    new TensorObjectArg(x),
                                                    new TensorObjectArg(y),

                                                };}
        std::string getReturnType() const override {return "float";}
        std::string getFunctionName() const override {return "sdot_mkl_internal";}

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject s;
        IndexVar i;
};

class MklAdd : public AbstractFunctionInterface{
    public: 
        MklAdd() : x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), CSR)),
                  y(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), CSR)),
                  z(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), CSR)),
                  i(IndexVar()) {};

        // IndexExpr getRHS() const override {return x(i);}
        // IndexExpr getLHS() const override {return x(i);}
        AcceleratorStmt getStmt() const override {return z(i, j) = x(i, j) + y(i, j);}
        std::vector<Argument> getArguments() const override {return 
                                                {   new DimArg(i), 
                                                    new TensorName(x),
                                                    new TensorName(y),
                                                    new TensorName(z)
                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "mkl_sparse_s_add_internal";}

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject z;
        IndexVar i;
        IndexVar j;
};

#endif 