#include "test.h"
#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_notation/accelerator_notation_nodes.h"

#include "op_factory.h"


using namespace taco;

TEST(accelerateNotation, AcceleratorExprNode) {
    // AcceleratorExpr((TensorVar()));
    // AcceleratorExpr expr = TensorVar();
    std::cout << AcceleratorExpr((TensorObject())) << std::endl;
    IndexVar i("i");

    TensorObject t("t", Type(taco::Float32, {Dimension()}));
    TensorObject x("x", Type(taco::Float32, {Dimension()}));

    std::cout << t(i) << endl;

    std::cout << AcceleratorExpr(0) << endl;
    std::cout << AcceleratorExpr(2.3) << endl;

    AcceleratorStmt stmt = t(i) = x(i);

    std::cout << stmt << std::endl;

}