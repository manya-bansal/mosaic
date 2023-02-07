#ifndef GSL_TENSOR_INTERFACE_H
#define GSL_TENSOR_INTERFACE_H


#include "taco/index_notation/index_notation.h"
#include "taco/type.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/accelerator_notation/accelerator_notation.h"

using namespace taco; 

class GslTensorPlus : public AbstractFunctionInterface{
    public: 
        GslTensorPlus() : 
                    x(TensorObject(Type(taco::Float32, {Dimension(), Dimension(), Dimension()}),  Format{Dense, Dense, Dense})),
                    y(TensorObject(Type(taco::Float32, {Dimension(), Dimension(), Dimension()}),  Format{Dense, Dense, Dense})),
                    i(IndexVar()),
                    j(IndexVar()),
                    k(IndexVar()),
                    l(IndexVar()),
                    var(DeclVar("tensor_float *", "var1")),
                    var2(DeclVar("tensor_float *", "var2")) {};

        AcceleratorStmt getStmt() const override {return y(i, j, k) = y(i, j, k) + x(i, j, k);}
        std::vector<Argument> getArguments() const override {
                                                return 
                                                {   new DeclVarArg(var2),
                                                    new DeclVarArg(var),
                                                }; }

        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "tensor_float_add";}

        std::vector<Argument>  callBefore() const override {
            taco::TransferLoad tensor_float_alloc("tensor_float_alloc", "tensor_float *");
            taco::TransferLoad set_tensor_data_s("set_tensor_data_s", "void");

            return {
                var = tensor_float_alloc(3, Dim(i)),
                var2 = tensor_float_alloc(3, Dim(i)),
                set_tensor_data_s(var, x),
                set_tensor_data_s(var2, y),
            };
        }

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject z;
        IndexVar i;
        IndexVar j;
        IndexVar k;
        IndexVar l;
        DeclVar var;
        DeclVar var2;

};
#endif