#ifndef STARDUST_INTERFACE_H
#define STARDUST_INTERFACE_H


#include "taco/index_notation/index_notation.h"
#include "taco/type.h"
#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/accelerator_notation/accel_interface.h"
#include <fstream>


static std::vector<std::string> customSplit(std::string str, char separator) {
    std::vector<std::string> strings;
    int startIndex = 0, endIndex = 0;
    for (size_t i = 0; i <= str.size(); i++) {
        
        // If we reached the end of the word or the end of the input.
        if (str[i] == separator || i == str.size()) {
            endIndex = i;
            std::string temp;
            temp.append(str, startIndex, endIndex - startIndex);
            strings.push_back(temp);
            startIndex = endIndex + 1;
        }
    }
    return strings;
}


static std::string stardustDataDir = "/home/reviewer/mosaic-benchmarks/stardust-runs/"; 

class StardustAdd : public AbstractFunctionInterface{
    public: 
        StardustAdd(const int& dim, const float& sparisty) : x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), CSR)),
                  y(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), CSR)),
                  z(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), CSR)),
                  i(IndexVar()),
                  j(IndexVar()),
                  dim(dim),
                  sparisty(sparisty) {};

        // IndexExpr getRHS() const override {return x(i);}
        // IndexExpr getLHS() const override {return x(i);}
        AcceleratorStmt getStmt() const override {return z(i, j) = x(i, j) + y(i, j);}
        std::vector<Argument> getArguments() const override {
            std::ifstream infile(stardustDataDir + "spmv_plus2.csv");
            std::string line;
            std::cout << "Begin" << std::endl;
            while (getline(infile, line,'\n')){
                for (auto word : customSplit(line, ',')){
                    std::cout << word << std::endl;
                }
            }
            throw std::exception();
            return
                { };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "nanosleep";}
        DynamicStmt getConstraints() const override {return (DynamicExpr(i) * DynamicExpr(j)) < 65536;}

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject z;
        IndexVar i;
        IndexVar j;
        int dim;
        float sparisty;
};


class StardustSpmv : public AbstractFunctionInterface{
    public: 
        StardustSpmv(const int& dim, const float& sparisty) : x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), Format{Dense, Sparse})), 
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  s(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()),
                  j(IndexVar()),
                  dim(dim),
                  sparisty(sparisty) {};
        AcceleratorStmt getStmt() const override {return y(i) = x(i, j)*s(j);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new TensorName(x),
                                                    new TensorObjectArg(s),
                                                    new TensorObjectArg(y),
                                                    new DimArg(i),
                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "nanosleep";}
         DynamicStmt getConstraints() const override {return (DynamicExpr(i) * DynamicExpr(j)) < 65536;}
    private: 
        TensorObject x;
        TensorObject y;
        TensorObject s;
        IndexVar i;
        IndexVar j;
        int dim;
        float sparisty;
};



#endif