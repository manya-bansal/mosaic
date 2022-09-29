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

        // std::vector<Argument>  callAfter() const override {
        //                 taco::TransferLoad call("callAfter", "custom_type");
        //                 return { var2 = call(x, y, var2, var, StringLiteral("\"ijk\""), DataArray(y), AddrDeclVarArg(var)) };
        //             }

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

#endif 
    