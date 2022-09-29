#ifndef TBLIS_INTERFACE_H
#define TBLIS_INTERFACE_H

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
                                taco::TransferLoad tblis_init_tensor_s("tblis_init_tensor_s", "void");
                                return { tblis_init_tensor_s(AddrDeclVarArg(var), 2, CastArg(new DimList(x), "len_type *"), DataArray(x), StringLiteral(" (stride_type[]) {1, 1}")),
                                         tblis_init_tensor_s(AddrDeclVarArg(var2), 2, CastArg(new DimList(y), "len_type *"), DataArray(y), StringLiteral("(stride_type[]) {1, 1}")), 
                                         tblis_init_tensor_s(AddrDeclVarArg(result), 2, CastArg(new DimList(x), "len_type *"), DataArray(z), StringLiteral("(stride_type[]) {1, 1}"))};
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
                                taco::TransferLoad tblis_init_tensor_s("tblis_init_tensor_s", "void");
                                return { tblis_init_tensor_s(AddrDeclVarArg(var), 3, CastArg(new DimList(y), "len_type *"), DataArray(y), StringLiteral(" (stride_type[]) {1, 1, 1}")),
                                         tblis_init_tensor_s(AddrDeclVarArg(var2), 2, CastArg(new DimList(z), "len_type *"), DataArray(z), StringLiteral("(stride_type[]) {1, 1}")), 
                                         tblis_init_tensor_s(AddrDeclVarArg(result), 3, CastArg(new DimList(y), "len_type *"), DataArray(x), StringLiteral("(stride_type[]) {1, 1, 1}"))};
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
                                return { tblis_init_vector_s(AddrDeclVarArg(var), CastArg(new DimArg(i), "len_type"), DataArray(x), CastArg(new LiteralArg(Datatype(taco::UInt32), 1), "stride_type")),
                                         tblis_init_vector_s(AddrDeclVarArg(var2), CastArg(new DimArg(i), "len_type"), DataArray(y), CastArg(new LiteralArg(Datatype(taco::UInt32), 1), "stride_type")),
                                         tblis_init_scalar_s(AddrDeclVarArg(result), 0)};
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

#endif 
    