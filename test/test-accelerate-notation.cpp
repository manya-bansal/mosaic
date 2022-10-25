#include "test.h"
#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_notation/accelerator_notation_nodes.h"
#include "taco/accelerator_notation/code_gen_dynamic_order.h"
#include "op_factory.h"


using namespace taco;


static TensorObject as("a", Float64), bs("b", Float64), cs("c", Float64),
                 ds("d", Float64), es("e", Float64);

static Type vectorType(Float64, {3});
static TensorObject a("a", vectorType), b("b", vectorType), c("c", vectorType),
                 d("d", vectorType), e("e", vectorType), f("f", vectorType);

static Type matrixType(Float64, {3,3});
static TensorObject A("A", matrixType), B("B", matrixType), C("C", matrixType);

static const IndexVar i("i"), j("j"), k("k");


TEST(accelerateNotation, AcceleratorExprTest) {
    // AcceleratorExpr((TensorVar()));
    // AcceleratorExpr expr = TensorVar();
    std::cout << AcceleratorExpr((TensorObject())) << std::endl;
    IndexVar i("i");

    TensorObject t("t", Type(taco::Float32, {Dimension()}));
    TensorObject x("x", Type(taco::Float32, {Dimension()}));
    TensorObject y("y", Type(taco::Float32));

    std::cout << t(i) << endl;

    std::cout << AcceleratorExpr(0) << endl;
    std::cout << AcceleratorExpr(2.3) << endl;

    AcceleratorStmt stmt = y += -x(i) / x(i) * x(i);

    std::cout << y - x(i) / x(i) * x(i) << std::endl;
    std::cout << forall(i, stmt) << std::endl;

    std::cout << sum(i, x(i)*t(i)) + 5 << endl;
    std::vector<IndexObject> incides = {new DynamicOrder(), new IndexVar(i)};

    DynamicOrder dynamicOrder;
    DynamicIndexIterator interator(dynamicOrder);
    DynamicIndexIterator interator2(dynamicOrder);
    
    IndexVar var; 
    IndexVar var2; 

    std::cout << (interator == (dynamicOrder(interator) + 1)) << endl;
    std::cout << (dynamicOrder(interator) == dynamicOrder(interator)) << endl;
    std::cout << (interator != interator) << endl;
    std::cout << (interator > interator) << endl;
    std::cout << (interator > (dynamicOrder(interator) + 1)) << endl;
    std::cout << (interator < (dynamicOrder(interator) + 1)) << endl;
    std::cout << (interator <= (dynamicOrder(interator) + 1)) << endl;
    std::cout << (interator >= (dynamicOrder(interator) + 1)) << endl;
    std::cout << forall(interator, (interator >= (dynamicOrder(interator) + 1))) << endl;
    std::cout << exists(interator, (interator >= (dynamicOrder(interator) + 1))) << endl;
    std::cout << forall(interator2, exists(interator, (interator >= DynamicExpr(var) + 1))) << endl;

    // GenerateSMTCode condition(DynamicExpr(var2) >= DynamicExpr(var) + 1, {});
    // cout << condition.generatePythonCode() << endl;

    // GenerateSMTCode condition2((DynamicExpr(var2) >= DynamicExpr(var) + 1) && (DynamicExpr(var2) >= DynamicExpr(var) + 1), {});
    // cout << condition2.generatePythonCode() << endl;
    std::map<DynamicOrder, std::vector<IndexVar>> mapRef; 
    std::map<IndexVar, int> dimRef;
    mapRef[dynamicOrder] = {var, var2, var};
    dimRef[var] = 10;
    dimRef[var2] = 10;
    GenerateSMTCode condition(forall(interator, dynamicOrder(interator) > 4), mapRef, dimRef, true);
    condition.runSMT();
    ASSERT_TRUE(condition.isSat());

    GenerateSMTCode condition1(forall(interator, interator > interator), mapRef, dimRef, true);
    condition1.runSMT();
    ASSERT_FALSE(condition1.isSat());

}

TEST(accelerateNotation, makeReductionNotation) {
  ASSERT_NOTATION_EQ(as = bs*cs,    makeReductionNotation(as = bs*cs));
  ASSERT_NOTATION_EQ(as = bs*cs*ds, makeReductionNotation(as = bs*cs*ds));
  ASSERT_NOTATION_EQ(as = bs+ds,    makeReductionNotation(as = bs+ds));
  ASSERT_NOTATION_EQ(as = bs-ds,    makeReductionNotation(as = bs-ds));

  ASSERT_NOTATION_EQ(as = sum(i, b(i)*c(i)),
                     makeReductionNotation(as = b(i)*c(i)));
  ASSERT_NOTATION_EQ(as = sum(i, b(i)*c(i)*d(i)),
                     makeReductionNotation(as=b(i)*c(i)*d(i)));
  ASSERT_NOTATION_EQ(as = sum(i, sum(j, b(i)*c(j))),
                     makeReductionNotation(as=b(i)*c(j)));
  ASSERT_NOTATION_EQ(as = sum(i, sum(j, sum(k, b(i)*c(j)*d(k)))),
                     makeReductionNotation(as=b(i)*c(j)*d(k)));

  ASSERT_NOTATION_EQ(as = sum(i, b(i)) + sum(j, c(j)),
                     makeReductionNotation(as=b(i)+c(j)));
  ASSERT_NOTATION_EQ(as = sum(i, b(i)) + sum(i, c(i)),
                     makeReductionNotation(as=b(i)+c(i)));
  ASSERT_NOTATION_EQ(as = sum(i, b(i)*c(i)) + sum(j, d(j)*e(j)),
                     makeReductionNotation(as = b(i)*c(i) + d(j)*e(j)));
  ASSERT_NOTATION_EQ(as = sum(i, b(i)*c(i)) + sum(i, sum(j, d(i)*e(j))),
                     makeReductionNotation(as = b(i)*c(i) + d(i)*e(j)));
  ASSERT_NOTATION_EQ(f(i) = b(i)*c(i),
                     makeReductionNotation(f(i)=b(i)*c(i)));
  ASSERT_NOTATION_EQ(as = sum(i, sum(j, B(i,j)*C(i,j) )),
                     makeReductionNotation(as = B(i,j)*C(i,j)));
  ASSERT_NOTATION_EQ(a(i) = sum(j, B(i,j)*c(j)),
                     makeReductionNotation(a(i)=B(i,j)*c(j)));
}