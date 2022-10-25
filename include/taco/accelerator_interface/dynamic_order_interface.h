#ifndef DYNAMIC_ORDER_INTERFACE_H
#define DYNAMIC_ORDER_INTERFACE_H


#include "taco/index_notation/index_notation.h"
#include "taco/type.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/accelerator_notation/accelerator_notation.h"


//tensor add with no reduction variables 
class TensorAdd : public AbstractFunctionInterface{
    public: 
        TensorAdd() : x(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()),
                  I(DynamicOrder()) {};

        taco::AcceleratorStmt getStmt() const override{ std::vector<IndexObject> indices = {new IndexVar(i), new DynamicOrder(I)};
                                                        return x[indices] = x[indices] + y[indices];}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override{return "test";}
        bool checkerFunction(IndexStmt stmt) const override{return true;}

    private: 
        TensorObject x;
        TensorObject y;
        IndexVar i;
        DynamicOrder I;
};

//tensor add with no reduction variables where all indices need to be of the same size
class TensorSquareAdd : public AbstractFunctionInterface{
    public: 
        TensorSquareAdd() : x(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()),
                  I(DynamicOrder()) {};

        taco::AcceleratorStmt getStmt() const override{ std::vector<IndexObject> indices = {new IndexVar(i), new DynamicOrder(I)};
                                                return x[indices] = x[indices] + y[indices];}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override{return "test";}
        bool checkerFunction(IndexStmt stmt) const override{return true;}
        DynamicStmt getConstraints() const override{ DynamicIndexIterator it(I);
                                       return forall(it, DynamicExpr(i) == I(it)); }

    private: 
        TensorObject x;
        TensorObject y;
        IndexVar i;
        DynamicOrder I;
};


//tensor add with no reduction variables where all indices need to be of the same size
class TensorReduxAdd : public AbstractFunctionInterface{
    public: 
        TensorReduxAdd() : x(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()),
                  I(DynamicOrder()) {};

        taco::AcceleratorStmt getStmt() const override{ std::vector<IndexObject> indicesRHS = {new IndexVar(i), new DynamicOrder(I)};
                                                        std::vector<IndexObject> indicesLHS = {new IndexVar(i), new DynamicOrder(I)};
                                                return x[indicesLHS] = x[indicesRHS] + y[indicesRHS];}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override{return "test";}
        bool checkerFunction(IndexStmt stmt) const override{return true;}

    private: 
        TensorObject x;
        TensorObject y;
        IndexVar i;
        DynamicOrder I;
};

//tensor add with no reduction variables where all indices need to be of the same size
class TensorSqaureReduxAdd : public AbstractFunctionInterface{
    public: 
        TensorSqaureReduxAdd() : x(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()),
                  I(DynamicOrder()) {};

        taco::AcceleratorStmt getStmt() const override{ std::vector<IndexObject> indicesRHS = {new IndexVar(i), new DynamicOrder(I)};
                                                        std::vector<IndexObject> indicesLHS = {new IndexVar(i), new DynamicOrder(I)};
                                                return x[indicesLHS] = x[indicesRHS] + y[indicesRHS];}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override{return "test";}
        bool checkerFunction(IndexStmt stmt) const override{return true;}
        DynamicStmt getConstraints() const override{ DynamicIndexIterator it(I);
                                       return forall(it, DynamicExpr(i) == I(it)); }

    private: 
        TensorObject x;
        TensorObject y;
        IndexVar i;
        DynamicOrder I;
};



#endif
