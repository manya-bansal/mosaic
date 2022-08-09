#include "test.h"
#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/index_notation/index_notation.h"

#include "op_factory.h"


using namespace taco;

TEST(accelerateNotation, AcceleratorExprNode) {
    // AcceleratorExpr((TensorVar()));
    // AcceleratorExpr expr = TensorVar();
    std::cout << AcceleratorExpr((TensorObject())) << std::endl;
    IndexVar i("i");

    TensorObject t("t", Type(taco::Float32, {Dimension()}));
    std::cout << t(i) << endl;

    std::cout << AcceleratorLiteral(0) << endl;
    std::cout << AcceleratorLiteral(2.3) << endl;

}