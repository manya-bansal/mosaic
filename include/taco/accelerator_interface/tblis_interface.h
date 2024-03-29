#ifndef TBLIS_INTERFACE_H
#define TBLIS_INTERFACE_H

#include "taco/index_notation/index_notation.h"
#include "taco/type.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/accelerator_notation/accelerator_notation.h"

class TblisMultiply : public AbstractFunctionInterface{
    public: 
        TblisMultiply() : 
                    x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}),  Format{Dense, Dense})),
                    y(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}),  Format{Dense, Dense})),
                    z(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}),  Format{Dense, Dense})),
                    i(IndexVar()),
                    j(IndexVar()),
                    k(IndexVar()),
                    var(DeclVar("tblis_tensor", "var1")),
                    var2(DeclVar("tblis_tensor", "var2")),
                    result(DeclVar("tblis_tensor", "result")) {};

        AcceleratorStmt getStmt() const override {return z(i, k) = x(i, j) * y(j, k);}
        std::vector<Argument> getArguments() const override {
                                                return 
                                                {   new StringLiteral("NULL"),
                                                    new StringLiteral("NULL"),
                                                    new AddrDeclVarArg(var),
                                                    new StringLiteral("\"ij\""),
                                                    new AddrDeclVarArg(var2),
                                                    new StringLiteral("\"jk\""),
                                                    new AddrDeclVarArg(result),
                                                    new StringLiteral("\"ik\"")
                                                }; }

        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "tblis_tensor_mult";}
        std::vector<Argument>  callBefore() const override {
                                taco::TransferLoad tblis_init_tensor_s_helper_row_major("tblis_init_tensor_s_helper_row_major", "void");
                                return { tblis_init_tensor_s_helper_row_major(AddrDeclVarArg(var), DimList(y), 2,  DataArray(x)),
                                         tblis_init_tensor_s_helper_row_major(AddrDeclVarArg(var2), DimList(y), 2,  DataArray(y)), 
                                         tblis_init_tensor_s_helper_row_major(AddrDeclVarArg(result), DimList(y), 2, DataArray(z)),
                                         };

                            }

        std::vector<Argument>  callAfter() const override {
            taco::TransferLoad free_tblis_tensor("free_tblis_tensor", "void");
                                return { 
                                        free_tblis_tensor(AddrDeclVarArg(var)),
                                        free_tblis_tensor(AddrDeclVarArg(var2)),
                                        free_tblis_tensor(AddrDeclVarArg(result))
                                        };

        }

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject z;
        IndexVar i;
        IndexVar j;
        IndexVar k;
        DeclVar var;
        DeclVar var2;
        DeclVar result;

};

class TblisTTM : public AbstractFunctionInterface{
    public: 
        TblisTTM() : 
                    x(TensorObject(Type(taco::Float32, {Dimension(), Dimension(), Dimension()}),  Format{Dense, Dense, Dense})),
                    y(TensorObject(Type(taco::Float32, {Dimension(), Dimension(), Dimension()}),  Format{Dense, Dense, Dense})),
                    z(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}),  Format{Dense, Dense})),
                    i(IndexVar()),
                    j(IndexVar()),
                    k(IndexVar()),
                    l(IndexVar()),
                    var(DeclVar("tblis_tensor", "var1")),
                    var2(DeclVar("tblis_tensor", "var2")),
                    result(DeclVar("tblis_tensor", "result")) {};

        AcceleratorStmt getStmt() const override {return x(i, j, k) = y(i, j, l) * z(k, l);}
        std::vector<Argument> getArguments() const override {
                                                return 
                                                {   new StringLiteral("NULL"),
                                                    new StringLiteral("NULL"),
                                                    new AddrDeclVarArg(var),
                                                    new StringLiteral("\"ijl\""),
                                                    new AddrDeclVarArg(var2),
                                                    new StringLiteral("\"kl\""),
                                                    new AddrDeclVarArg(result),
                                                    new StringLiteral("\"ijk\"")
                                                }; }

        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "tblis_tensor_mult";}
        std::vector<Argument>  callBefore() const override {
                                taco::TransferLoad tblis_init_tensor_s_helper_row_major("tblis_init_tensor_s_helper_row_major", "void");
                                return { tblis_init_tensor_s_helper_row_major(AddrDeclVarArg(var), DimList(y), 3,  DataArray(y)),
                                         tblis_init_tensor_s_helper_row_major(AddrDeclVarArg(var2), DimList(z), 2,  DataArray(z)), 
                                         tblis_init_tensor_s_helper_row_major(AddrDeclVarArg(result), DimList(y), 3, DataArray(x))};
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
        DeclVar result;

};

class TblisDot : public AbstractFunctionInterface{
    public: 
        TblisDot() :x(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                    y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                    s(TensorObject(Type(taco::Float32))),
                    i(IndexVar()),
                    var(DeclVar("tblis_vector", "var1")),
                    var2(DeclVar("tblis_vector", "var2")),
                    result(DeclVar("tblis_scalar", "result")) {};

        AcceleratorStmt getStmt() const override {return s = x(i) * y(i);}
        std::vector<Argument> getArguments() const override {
                                                return 
                                                {   new StringLiteral("NULL"),
                                                    new StringLiteral("NULL"),
                                                    new AddrDeclVarArg(var),
                                                    new AddrDeclVarArg(var2),
                                                    new AddrDeclVarArg(result)
                                                }; }

        std::string getReturnType() const override {return "float";}
        std::string getFunctionName() const override {return "tblis_vector_dot_transfer";}
        std::vector<Argument>  callBefore() const override {
                                taco::TransferLoad tblis_init_vector_s("tblis_init_vector_s", "void");
                                taco::TransferLoad tblis_init_scalar_s("tblis_init_scalar_s", "void");
                                taco::TransferLoad print_done("printf", "void");
                                return { tblis_init_vector_s(AddrDeclVarArg(var), CastArg(new DimArg(i), "len_type"), DataArray(x), CastArg(new LiteralArg(Datatype(taco::UInt32), 1), "stride_type")),
                                         tblis_init_vector_s(AddrDeclVarArg(var2), CastArg(new DimArg(i), "len_type"), DataArray(y), CastArg(new LiteralArg(Datatype(taco::UInt32), 1), "stride_type")),
                                         tblis_init_scalar_s(AddrDeclVarArg(result), 0)
                                        //  print_done("done")
                                        
                                        };
                            }

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject s;
        IndexVar i;
        DeclVar var;
        DeclVar var2;
        DeclVar result;

};


class TblisSaxpy : public AbstractFunctionInterface{
    public: 
        TblisSaxpy() :x(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                    y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                    s(TensorObject(Type(taco::Float32))),
                    i(IndexVar()),
                    var(DeclVar("tblis_vector", "var1")),
                    var2(DeclVar("tblis_vector", "var2")),
                    result(DeclVar("tblis_scalar", "result")) {};

        AcceleratorStmt getStmt() const override {return y(i) = x(i) + y(i);}
        std::vector<Argument> getArguments() const override {
                                                return 
                                                {   new StringLiteral("NULL"),
                                                    new StringLiteral("NULL"),
                                                    new AddrDeclVarArg(var),
                                                    new AddrDeclVarArg(var2)
                                                }; }

        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "tblis_vector_add";}
        std::vector<Argument>  callBefore() const override {
                                taco::TransferLoad tblis_init_vector_s("tblis_init_vector_s", "void");
                                taco::TransferLoad tblis_init_scalar_s("tblis_init_scalar_s", "void");
                                return { tblis_init_vector_s(AddrDeclVarArg(var), CastArg(new DimArg(i), "len_type"), DataArray(x), CastArg(new LiteralArg(Datatype(taco::UInt32), 1), "stride_type")),
                                         tblis_init_vector_s(AddrDeclVarArg(var2), CastArg(new DimArg(i), "len_type"), DataArray(y), CastArg(new LiteralArg(Datatype(taco::UInt32), 1), "stride_type"))};
                            }

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject s;
        IndexVar i;
        DeclVar var;
        DeclVar var2;
        DeclVar result;

};


class TblisPlus : public AbstractFunctionInterface{
    public: 
        TblisPlus() : 
                    x(TensorObject(Type(taco::Float32, {Dimension(), Dimension(), Dimension()}),  Format{Dense, Dense, Dense})),
                    y(TensorObject(Type(taco::Float32, {Dimension(), Dimension(), Dimension()}),  Format{Dense, Dense, Dense})),
                    z(TensorObject(Type(taco::Float32, {Dimension(), Dimension(), Dimension()}),  Format{Dense, Dense, Dense})),
                    i(IndexVar()),
                    j(IndexVar()),
                    k(IndexVar()),
                    l(IndexVar()),
                    var(DeclVar("tblis_tensor", "var1")),
                    var2(DeclVar("tblis_tensor", "var2")),
                    result(DeclVar("tblis_tensor", "result")) {};

        AcceleratorStmt getStmt() const override {return x(i, j, k) = y(i, j, k) + x(i, j, k);}
        std::vector<Argument> getArguments() const override {
                                                return 
                                                {   new StringLiteral("NULL"),
                                                    new StringLiteral("NULL"),
                                                    new AddrDeclVarArg(var2),
                                                    new StringLiteral("\"ijk\""),
                                                    new AddrDeclVarArg(var),
                                                    new StringLiteral("\"ijk\""),
                                                }; }

        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "tblis_tensor_add";}
        std::vector<Argument>  callBefore() const override {
                                taco::TransferLoad tblis_init_tensor_s_helper_row_major("tblis_init_tensor_s_helper_row_major", "void");
                                return { tblis_init_tensor_s_helper_row_major(AddrDeclVarArg(var), DimList(y), 3,  DataArray(x)),
                                         tblis_init_tensor_s_helper_row_major(AddrDeclVarArg(var2), DimList(y), 3,  DataArray(y)),};
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
        DeclVar result;

};

class TblisGemv : public AbstractFunctionInterface{
    public: 
        TblisGemv() : 
                    x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}),  Format{Dense, Dense})),
                    y(TensorObject(Type(taco::Float32, {Dimension()}),  Format{Dense})),
                    z(TensorObject(Type(taco::Float32, {Dimension()}),  Format{Dense})),
                    i(IndexVar()),
                    j(IndexVar()),
                    k(IndexVar()),
                    var(DeclVar("tblis_tensor", "var1")),
                    var2(DeclVar("tblis_tensor", "var2")),
                    result(DeclVar("tblis_tensor", "result")) {};

        AcceleratorStmt getStmt() const override {return z(i) = x(i, j) * y(j);}
        std::vector<Argument> getArguments() const override {
                                                return 
                                                {   new StringLiteral("NULL"),
                                                    new StringLiteral("NULL"),
                                                    new AddrDeclVarArg(var),
                                                    new StringLiteral("\"ij\""),
                                                    new AddrDeclVarArg(var2),
                                                    new StringLiteral("\"j\""),
                                                    new AddrDeclVarArg(result), 
                                                    new StringLiteral("\"i\"")
                                                }; }

        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "tblis_tensor_mult";}
        std::vector<Argument>  callBefore() const override {
                                taco::TransferLoad tblis_init_tensor_s_helper_row_major("tblis_init_tensor_s_helper_row_major", "void");
                                taco::TransferLoad tblis_set_vector("tblis_set_vector", "void");
                                return { tblis_init_tensor_s_helper_row_major(AddrDeclVarArg(var), DimList(x), 2,  DataArray(x)),
                                         tblis_set_vector(AddrDeclVarArg(var2), DimList(y), 1,  DataArray(y)), 
                                         tblis_set_vector(AddrDeclVarArg(result), DimList(y), 1, DataArray(z))
                                         };

                            }

        std::vector<Argument>  callAfter() const override {
            taco::TransferLoad free_tblis_tensor("free_tblis_tensor", "void");
                                return { 
                                        free_tblis_tensor(AddrDeclVarArg(var)),
                                        free_tblis_tensor(AddrDeclVarArg(var2)),
                                        free_tblis_tensor(AddrDeclVarArg(result))
                                        };

        }

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject z;
        IndexVar i;
        IndexVar j;
        IndexVar k;
        DeclVar var;
        DeclVar var2;
        DeclVar result;

};

class TblisTTV : public AbstractFunctionInterface{
    public: 
        TblisTTV() : 
                    x(TensorObject(Type(taco::Float32, {Dimension(), Dimension(), Dimension()}),  Format{Dense, Dense, Dense})),
                    y(TensorObject(Type(taco::Float32, {Dimension()}),  Format{Dense})),
                    z(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}),  Format{Dense, Dense})),
                    i(IndexVar()),
                    j(IndexVar()),
                    k(IndexVar()),
                    l(IndexVar()),
                    var(DeclVar("tblis_tensor", "var1")),
                    var2(DeclVar("tblis_tensor", "var2")),
                    result(DeclVar("tblis_tensor", "result")) {};

        AcceleratorStmt getStmt() const override {return z(i, j) = x(i, j, k) * y(k);}
        std::vector<Argument> getArguments() const override {
                                                return 
                                                {   new StringLiteral("NULL"),
                                                    new StringLiteral("NULL"),
                                                    new AddrDeclVarArg(var),
                                                    new StringLiteral("\"ijk\""),
                                                    new AddrDeclVarArg(var2),
                                                    new StringLiteral("\"k\""),
                                                    new AddrDeclVarArg(result),
                                                    new StringLiteral("\"ij\"")
                                                }; }

        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "tblis_tensor_mult";}
        std::vector<Argument>  callBefore() const override {
                                taco::TransferLoad tblis_init_tensor_s_helper_row_major_dim("tblis_init_tensor_s_helper_row_major_dim", "void");
                                return { tblis_init_tensor_s_helper_row_major_dim(AddrDeclVarArg(var), Dim(i), 3,  DataArray(y)),
                                         tblis_init_tensor_s_helper_row_major_dim(AddrDeclVarArg(var2), Dim(i), 1,  DataArray(z)), 
                                         tblis_init_tensor_s_helper_row_major_dim(AddrDeclVarArg(result), Dim(i), 2, DataArray(x))};
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
        DeclVar result;

};

#endif 
    