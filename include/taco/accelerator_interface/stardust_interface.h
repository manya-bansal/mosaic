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
        StardustAdd(const std::string& name) : x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), CSR)),
                  y(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), CSR)),
                  z(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), CSR)),
                  i(IndexVar()),
                  j(IndexVar()),
                  name(name) {};

        // IndexExpr getRHS() const override {return x(i);}
        // IndexExpr getLHS() const override {return x(i);}
        AcceleratorStmt getStmt() const override {return z(i, j) = x(i, j) + y(i, j);}
        std::vector<Argument> getArguments() const override {
            std::ifstream infile(stardustDataDir + "spmv_plus2.csv");
            std::string line;
            while (getline(infile, line,'\n')){
                std::vector<std::string> words = customSplit(line, ',');
                if (words.size() < 9){
                    continue;
                }

                if (words[2] == "Plus2CSR"){
                    if (words[4] == name){
                        return {
                            new LiteralArg(Datatype(taco::UInt32), stoi(words[6]))
                        };
                    }
                }
            }
            taco_uerror << "Tried to call a function that stardust has no data for";
            // Silence Warnings
            return {};
        }
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "nanosleep";}
        DynamicStmt getConstraints() const override {return (DynamicExpr(i) * DynamicExpr(j)) < 65536;}

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject z;
        IndexVar i;
        IndexVar j;
        std::string name;
};


class StardustSpmv : public AbstractFunctionInterface{
    public: 
        StardustSpmv(const std::string& name) : x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), Format{Dense, Sparse})), 
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  s(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()),
                  j(IndexVar()),
                  name(name) {};
        AcceleratorStmt getStmt() const override {return y(i) = x(i, j)*s(j);}
        std::vector<Argument> getArguments() const override {
            std::ifstream infile(stardustDataDir + "spmv_plus2.csv");
            std::string line;
            while (getline(infile, line,'\n')){
                std::vector<std::string> words = customSplit(line, ',');
                if (words.size() < 9){
                    continue;
                }

                if (words[2] == "SpMV"){
                    if (words[4] == name){
                        return {
                            new LiteralArg(Datatype(taco::UInt32), stoi(words[6]))
                        };
                    }
                }
            }
            taco_uerror << "Tried to call a function that stardust has no data for";
            return {};
        };
        
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "escape";}
    private: 
        TensorObject x;
        TensorObject y;
        TensorObject s;
        IndexVar i;
        IndexVar j;
        std::string name;
};



#endif