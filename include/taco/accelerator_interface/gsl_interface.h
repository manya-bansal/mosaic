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


#endif 