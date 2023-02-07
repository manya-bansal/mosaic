#ifndef GSL_INTERFACE_H
#define GSL_INTERFACE_H


#include "taco/index_notation/index_notation.h"
#include "taco/type.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/accelerator_notation/accelerator_notation.h"

class GSLVecAdd : public AbstractFunctionInterface{
    public: 
        GSLVecAdd() : x(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()),
                  a(DeclVar("gsl_vector_float * ", "a")),
                  b(DeclVar("gsl_vector_float *", "b")) {};

        taco::AcceleratorStmt getStmt() const override{ return x(i) = x(i) + y(i);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new DeclVarArg(a),
                                                    new DeclVarArg(b),
                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override{return "gsl_vector_float_add";}
        bool checkerFunction(IndexStmt stmt) const override{return true;}

        std::vector<Argument>  callBefore() const override {
            taco::TransferLoad gsl_vector_float_calloc("gsl_vector_float_calloc", "gsl_vector_float *");
            taco::TransferLoad set_gsl_float_data("set_gsl_float_data", "void");
            return {
                a = gsl_vector_float_calloc(Dim(i)),
                b = gsl_vector_float_calloc(Dim(i)),
                set_gsl_float_data(a, x),
                set_gsl_float_data(b, y)
            };
        }

    private: 
        TensorObject x;
        TensorObject y;
        IndexVar i;
        DeclVar a;
        DeclVar b;
};



class GSLDot : public AbstractFunctionInterface{
    public: 
        GSLDot() : x(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  s(TensorObject(Type(taco::Float32))),
                  i(IndexVar()),
                  a(DeclVar("gsl_vector_float * ", "a")),
                  b(DeclVar("gsl_vector_float *", "b")) {};

        taco::AcceleratorStmt getStmt() const override{ return s = x(i) * y(i);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new DeclVarArg(a),
                                                    new DeclVarArg(b),
                                                    new AddrTensorVar(s)

                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override{return "gsl_blas_sdot";}
        bool checkerFunction(IndexStmt stmt) const override{return true;}

        std::vector<Argument>  callBefore() const override {
            taco::TransferLoad gsl_vector_float_calloc("gsl_vector_float_calloc", "gsl_vector_float *");
            taco::TransferLoad set_gsl_float_data("set_gsl_float_data", "void");
            return {
                a = gsl_vector_float_calloc(Dim(i)),
                b = gsl_vector_float_calloc(Dim(i)),
                set_gsl_float_data(a, x),
                set_gsl_float_data(b, y)
            };
        }

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject s;
        IndexVar i;
        DeclVar a;
        DeclVar b;
};


class GSLSgemv : public AbstractFunctionInterface{
    public: 
        GSLSgemv() : x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), Format{Dense, Dense})), 
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  s(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()),
                  j(IndexVar()),
                  a(DeclVar("gsl_vector_float * ", "var1")),
                  b(DeclVar("gsl_vector_float *", "var2")),
                  mat(DeclVar("gsl_matrix_float *", "mat")) {}
        AcceleratorStmt getStmt() const override {return y(i) = x(i, j) * s(j) + y(i);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new StringLiteral("111"), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new DeclVarArg(mat),
                                                    new DeclVarArg(a),
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new DeclVarArg(b),

                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override{return "gsl_blas_sgemv";}
        bool checkerFunction(IndexStmt stmt) const override{return true;}

        std::vector<Argument>  callBefore() const override {
            taco::TransferLoad gsl_vector_float_calloc("gsl_vector_float_calloc", "gsl_vector_float *");
            taco::TransferLoad set_gsl_float_data("set_gsl_float_data", "void");
            taco::TransferLoad gsl_matrix_float_alloc("gsl_matrix_float_alloc", "gsl_matrix_float *");
            taco::TransferLoad set_gsl_mat_data_row_major_s("set_gsl_mat_data_row_major_s", "void");

            return {
                a = gsl_vector_float_calloc(Dim(j)),
                b = gsl_vector_float_calloc(Dim(i)),
                set_gsl_float_data(a, s),
                set_gsl_float_data(b, y),
                mat = gsl_matrix_float_alloc(Dim(i), Dim(j)),
                set_gsl_mat_data_row_major_s(mat, x)
            };
        }

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject s;
        IndexVar i;
        IndexVar j;
        DeclVar a;
        DeclVar b;
        DeclVar mat;
};

class GSLGemv : public AbstractFunctionInterface{
    public: 
        GSLGemv() : x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), Format{Dense, Dense})), 
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  s(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()),
                  j(IndexVar()),
                  a(DeclVar("gsl_vector_float * ", "var1")),
                  b(DeclVar("gsl_vector_float *", "var2")),
                  mat(DeclVar("gsl_matrix_float *", "mat")) {};
        AcceleratorStmt getStmt() const override {return y(i) = x(i, j) * s(j);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new StringLiteral("111"), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new DeclVarArg(mat),
                                                    new DeclVarArg(a),
                                                    new LiteralArg(Datatype(taco::UInt32), 0),
                                                    new DeclVarArg(b),

                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override{return "gsl_blas_sgemv";}
        bool checkerFunction(IndexStmt stmt) const override{return true;}

        std::vector<Argument>  callBefore() const override {
            taco::TransferLoad gsl_vector_float_calloc("gsl_vector_float_calloc", "gsl_vector_float *");
            taco::TransferLoad set_gsl_float_data("set_gsl_float_data", "void");
            taco::TransferLoad gsl_matrix_float_alloc("gsl_matrix_float_alloc", "gsl_matrix_float *");
            taco::TransferLoad set_gsl_mat_data_row_major_s("set_gsl_mat_data_row_major_s", "void");

            return {
                a = gsl_vector_float_calloc(Dim(j)),
                b = gsl_vector_float_calloc(Dim(i)),
                set_gsl_float_data(a, s),
                set_gsl_float_data(b, y),
                mat = gsl_matrix_float_alloc(Dim(i), Dim(j)),
                set_gsl_mat_data_row_major_s(mat, x)
            };
        }

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject s;
        IndexVar i;
        IndexVar j;
        DeclVar a;
        DeclVar b;
        DeclVar mat;
};


class GSLMM : public AbstractFunctionInterface{
    public: 
        GSLMM() : x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), Format{Dense, Dense})), 
                  y(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), Format{Dense, Dense})), 
                  z(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), Format{Dense, Dense})), 
                  i(IndexVar()),
                  j(IndexVar()),
                  k(IndexVar()),
                  a(DeclVar("gsl_matrix_float * ", "var1")),
                  b(DeclVar("gsl_matrix_float *", "var2")),
                  mat(DeclVar("gsl_matrix_float *", "mat")) {};
        AcceleratorStmt getStmt() const override {return z(i, k) = x(i, j) *  y(j, k);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new StringLiteral("111"), 
                                                    new StringLiteral("111"), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new DeclVarArg(mat),
                                                    new DeclVarArg(a),
                                                    new LiteralArg(Datatype(taco::UInt32), 0),
                                                    new DeclVarArg(b),

                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override{return "gsl_blas_sgemm";}
        bool checkerFunction(IndexStmt stmt) const override{return true;}

        std::vector<Argument>  callBefore() const override {
            taco::TransferLoad gsl_matrix_float_alloc("gsl_matrix_float_alloc", "gsl_matrix_float *");
            taco::TransferLoad set_gsl_mat_data_row_major_s("set_gsl_mat_data_row_major_s", "void");

            return {
                a = gsl_matrix_float_alloc(Dim(i), Dim(j)),
                b = gsl_matrix_float_alloc(Dim(j), Dim(k)),
                mat = gsl_matrix_float_alloc(Dim(i), Dim(k)),
                set_gsl_mat_data_row_major_s(mat, x),
                set_gsl_mat_data_row_major_s(a, y),
                set_gsl_mat_data_row_major_s(b, z)
            };
        }

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject z;
        IndexVar i;
        IndexVar j;
        IndexVar k;
        DeclVar a;
        DeclVar b;
        DeclVar mat;
};


class GSLSymmetricGemv : public AbstractFunctionInterface{
    public: 
        GSLSymmetricGemv() : x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), Format{Dense, Dense})), 
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  s(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()),
                  j(IndexVar()),
                  a(DeclVar("gsl_vector_float * ", "var1")),
                  b(DeclVar("gsl_vector_float *", "var2")),
                  mat(DeclVar("gsl_matrix_float *", "mat")) {};
        AcceleratorStmt getStmt() const override {
                                                    return y(i) = x(i, j) * s(j);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new StringLiteral("121"), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new DeclVarArg(mat),
                                                    new DeclVarArg(a),
                                                    new LiteralArg(Datatype(taco::UInt32), 0),
                                                    new DeclVarArg(b),

                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override{return "gsl_blas_ssymv";}
        bool checkerFunction(IndexStmt stmt) const override{return true;}

        std::vector<Argument>  callBefore() const override {
            taco::TransferLoad gsl_vector_float_calloc("gsl_vector_float_calloc", "gsl_vector_float *");
            taco::TransferLoad set_gsl_float_data("set_gsl_float_data", "void");
            taco::TransferLoad gsl_matrix_float_alloc("gsl_matrix_float_alloc", "gsl_matrix_float *");
            taco::TransferLoad set_gsl_mat_data_row_major_s("set_gsl_mat_data_row_major_s", "void");

            return {
                a = gsl_vector_float_calloc(Dim(j)),
                b = gsl_vector_float_calloc(Dim(i)),
                set_gsl_float_data(a, s),
                set_gsl_float_data(b, y),
                mat = gsl_matrix_float_alloc(Dim(i), Dim(j)),
                set_gsl_mat_data_row_major_s(mat, x)
            };
        }

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject s;
        IndexVar i;
        IndexVar j;
        DeclVar a;
        DeclVar b;
        DeclVar mat;
};


#endif 