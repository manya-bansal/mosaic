#include "test.h"
#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_notation/accelerator_notation_nodes.h"
#include "taco/accelerator_notation/code_gen_dynamic_order.h"
#include "taco/accelerator_interface/tile_interface.h"
#include "op_factory.h"
#include <chrono>
#include <unistd.h>
#include <random>

#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_interface/test_interface.h"
#include "taco/accelerator_interface/cuda_interface.h"
#include "taco/accelerator_interface/tile_interface.h"
#include "taco/accelerator_interface/tensorflow_interface.h"
#include "taco/accelerator_interface/gsl_interface.h"
#include "taco/accelerator_interface/tensor_interface.h"
#include "taco/accelerator_interface/avx2_interface.h"
#include "taco/accelerator_interface/dynamic_order_interface.h"
#include "taco/accelerator_interface/mkl_interface.h"

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


    std::cout << x[incides]<< std::endl;

    DynamicOrder dynamicOrder;
    DynamicIndexIterator interator(dynamicOrder);
    DynamicIndexIterator interator2(dynamicOrder);
    
    IndexVar var; 
    IndexVar var2; 

    std::cout << (interator == (dynamicOrder(interator) + 1)) << endl;


    std::cout << forall(interator,(dynamicOrder(interator)  == 4)) << endl;


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

    cout << util::join(condition.getTilings()) << endl;

    GenerateSMTCode condition1(forall(interator, interator > interator), mapRef, dimRef, true);
    condition1.runSMT();
    ASSERT_FALSE(condition1.isSat());

    //property notation test

    cout <<  (PropertyTag("symmetric") = (PropertyExpr("symmetric") + PropertyExpr("symmetric"))) << endl;


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



TEST(accelerateNotation, testCCLTimeAVX) {

    auto start = std::chrono::high_resolution_clock::now();

    DynamicOrder dynamicOrder;
    DynamicIndexIterator interator(dynamicOrder);
    IndexVar var;

    std::map<DynamicOrder, std::vector<IndexVar>> mapRef; 
    std::map<IndexVar, int> dimRef;

    mapRef[dynamicOrder] = {var};
    dimRef[var] = 10;

    GenerateSMTCode condition(forall(interator, dynamicOrder(interator) == 4), mapRef, dimRef, true);
    condition.runSMT();
    ASSERT_TRUE(condition.isSat());

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by function here: "
        << duration.count() << " us" << std::endl;


    condition.getTilings();

}

TEST(accelerateNotation, testCCLTimeStardust) {


    DynamicOrder dynamicOrder;
    DynamicIndexIterator interator(dynamicOrder);
    IndexVar var;

    std::map<DynamicOrder, std::vector<IndexVar>> mapRef; 
    std::map<IndexVar, int> dimRef;

    IndexVar var2, var3, var4, var5;
    dimRef[var] = 20;
    dimRef[var2] = 20;
    dimRef[var3] = 20;
    dimRef[var4] = 20;

    GenerateSMTCode condition( ((DynamicExpr(var) * DynamicExpr(var2) * DynamicExpr(var3) * DynamicExpr(var4)) == 65536),
    mapRef, dimRef, true);

    std::cout << util::join(condition.getTilings()) << std::endl;

}




bool checker_function(int tile){
  if (tile < 16) return true;
  return false;
}

int tryall(int num)
{
    for (int i = 0; i < num; i++){
      usleep(326);
    }
}

void findIndex(int last, IndexStmt stmt, std::vector<IndexVar> vars, IndexExpr accelerateExpr)
{   
  for (auto var: vars){
    for (int i = 0; i < last; i++){
      usleep(326);
      if (checker_function(i)){
        tryall(14);
        return;
      }
    }
  }
    
}

int randomSearch(int last){

  std::mt19937 rng(9);
  std::uniform_int_distribution<int> gen(1, last); // uniform, unbiased

  std:set<int> seen;

  int r = gen(rng);
  int i = 0;
  while (!checker_function(r)){
    i++;
    seen.insert(r);
    while (seen.count(r) == 1){
      r = gen(rng);
    }
  }
  std::cout << r << std::endl;
  return i;
}


TEST (accelerateNotation, testCheckerFunction){

  // binary search over 0-65536
  //   Tensor<float> A("A", {16}, Format{Dense});
  //  Tensor<float> expected("expected", {16}, Format{Dense});
  //  Tensor<float> B("B", {16}, Format{Dense});
  //  Tensor<float> C("C", {16}, Format{Dense});
  //  IndexVar i("i");

  //  for (int i = 0; i < 16; i++) {
  //     C.insert({i}, (float) i);
  //     B.insert({i}, (float) i);
  //  }

  //  C.pack();
  //  B.pack();

  //  IndexExpr accelerateExpr = B(i) + C(i);
  //  A(i) = accelerateExpr;
  //  IndexStmt stmt = A.getAssignment().concretize();
  //  stmt = stmt.tile(new TileSaxpy(), accelerateExpr, {{i,  4}});

  //   auto start = std::chrono::high_resolution_clock::now();
  //   findIndex(65536, stmt, {i}, accelerateExpr);
  //     auto stop = std::chrono::high_resolution_clock::now();

  //   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  //   std::cout << "Time taken by function: "
  //         << duration.count() << " us" << std::endl;
  
  auto start = std::chrono::high_resolution_clock::now();
  int i = randomSearch(65536);
  auto stop = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Time taken by function: "
        << duration.count() << " us" << std::endl;

  std::cout << i << std::endl;


}


TEST (accelerateNotation, timeEndToEnd){

  // binary search over 0-65536
   Tensor<float> A("A", {16, 16}, Format{Dense, Dense});
   Tensor<float> B("B", {16, 16}, Format{Dense, Dense});
   Tensor<float> C("C", {16, 16}, Format{Dense, Dense});
   Tensor<float> D("D", {16, 16}, CSR);
   IndexVar i("i");
   IndexVar j("j");
   IndexVar k("k");


  IndexExpr accelerateExpr = B(i, j) * C(j, k);
  A(i, k) = accelerateExpr;

  std::vector<FunctionInterface> interfaces =  {
                         new MatrixMultiply()
                          // new CudaSpmv(), new GSLVecAdd(), new MklSgemv(),
                          // new SparseMklSgemv(), new SparseMklMM(), new MklSymmgemv(),
                          // new MklMM(), new MklDot(), new MklAdd(), new SparseMklMMCOOCSR(), 
                          // new TblisMultiply(), new TblisTTM(), new TblisDot(), 
                          // new TblisSaxpy(), new TblisPlus(), new TblisGemv(), 
                          // new TblisTTV(), new GslTensorPlus()
  };

  //  std::vector<FunctionInterface> interfaces =  {
  //                         new MatrixMultiply()
  // };

  cout << "Registering #" << interfaces.size() << " Interfaces!!";

  A.registerAccelerators(interfaces);

  A.accelerateOn();

  A.compile();
  A.assemble();
  A.compute();


  // auto start = std::chrono::high_resolution_clock::now();
  // int i = randomSearch(65536);
  // auto stop = std::chrono::high_resolution_clock::now();

  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  // std::cout << "Time taken by function: "
  //       << duration.count() << " us" << std::endl;

  // std::cout << i << std::endl;


}

