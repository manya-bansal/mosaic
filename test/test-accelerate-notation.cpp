#include "test.h"
#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/accelerator_interface/tile_interface.h"
#include "taco/accelerator_interface/cblas_interface.h"
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


    std::cout << x[incides]<< std::endl;

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

    

    DynamicOrder dynamicOrder;
    DynamicIndexIterator interator(dynamicOrder);
    IndexVar var;

    std::map<DynamicOrder, std::vector<IndexVar>> mapRef; 
    std::map<IndexVar, int> dimRef;

    mapRef[dynamicOrder] = {var};
    dimRef[var] = 10;

    GenerateSMTCode condition(forall(interator, dynamicOrder(interator) == 4), mapRef, dimRef, true);

    // We have instrumented the runSMT function to report the time to run the 
    // query not generate it.
    condition.runSMT();
    ASSERT_TRUE(condition.isSat());

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

    // We have instrumented the runSMT function to report the time to run the 
    // query not generate it.
    condition.runSMT();
    ASSERT_TRUE(condition.isSat());

}

bool checker_function(int tile){
  if (tile < 16) return true;
  return false;
}


TEST (accelerateNotation, testRandomStardust){

  // Set up code for the computation we want to schedule. 

  Tensor<float> A("A", {16}, Format{Dense});
  Tensor<float> expected("expected", {16}, Format{Dense});
  Tensor<float> B("B", {16}, Format{Dense});
  Tensor<float> C("C", {16}, Format{Dense});
  IndexVar i("i");

  for (int i = 0; i < 16; i++) {
    C.insert({i}, (float) i);
    B.insert({i}, (float) i);
  }

  C.pack();
  B.pack();

  IndexExpr accelerateExpr = B(i) + C(i);
  A(i) = accelerateExpr;
  IndexStmt stmt = A.getAssignment().concretize();

  double count = 0;

  for (int j = 0; j < 10; j++){
    auto start = std::chrono::high_resolution_clock::now();
    // Starting the random Search
    std::mt19937 rng(time(nullptr));
    // uniform, unbiased. Bound by twice the capabilty of Stardust.
    std::uniform_int_distribution<int> gen(1, 65536*2); 

    int r1 = gen(rng);

    // Find at least one tiling that works.
    while (r1 > 65536){
      // Schedule the computation
      IndexStmt newStmt = stmt.tile(new TileSaxpy(), accelerateExpr, {{i,  r1}});
      r1 = gen(rng);
    }

    // Now start linear search to maximize tile size.
    while (r1 < 65536){
      // Schedule the computation
      IndexStmt newStmt = stmt.tile(new TileSaxpy(), accelerateExpr, {{i,  r1}});
      r1++;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    count += duration.count();
  }

  // Divide by 10 to average over 10 runs and then divide by 1000000
  // to convert microseconds into seconds.
  std::cout << "Time taken to Run Random Search for Stardust: "
          << float(count)/float(1000000 * 10) << "s" << std::endl;

}

bool check_stardust(int r1){
  if (r1 < 65536) return true;
  return false;

}
TEST (accelerateNotation, testLinearStardust){

  // Set up code for the computation we want to schedule. 
  Tensor<float> A("A", {16}, Format{Dense});
  Tensor<float> expected("expected", {16}, Format{Dense});
  Tensor<float> B("B", {16}, Format{Dense});
  Tensor<float> C("C", {16}, Format{Dense});
  IndexVar i("i");

  for (int i = 0; i < 16; i++) {
    C.insert({i}, (float) i);
    B.insert({i}, (float) i);
  }

  C.pack();
  B.pack();

  IndexExpr accelerateExpr = B(i) + C(i);
  A(i) = accelerateExpr;
  IndexStmt stmt = A.getAssignment().concretize();

  // Time how long it takes to schedule one time, we will delay to account for
  // the cost of computation while scheduling. 
  auto start1 = std::chrono::high_resolution_clock::now();
  IndexStmt newStmt = stmt.tile(new TileSaxpy(), accelerateExpr, {{i,  1}});
  auto end1 = std::chrono::high_resolution_clock::now();

  double count = 0;
  // for (int j = 0; j < 10; j++){
  auto start = std::chrono::high_resolution_clock::now();
  // Starting the random Search
  std::mt19937 rng(time(nullptr));
  // uniform, unbiased. Bound by twice the capabilty of Stardust.
  std::uniform_int_distribution<int> gen(1, 65536*2); 


  // Now start linear search.
  int r1 = 0;
  // Continue until to maximimum tile size is reached.
  while (check_stardust(r1)){
    // We delay to account for the time it takes to schedule the computation.
    // See up for timing logic.
    usleep(std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count());
    r1++;
  }

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  count = duration.count();
  // Dvide by 1000000 to convert microseconds into seconds.
  std::cout << "Time taken to Run Linear Search for Stardust: "
          << float(count)/float(1000000) << "s" << std::endl;
}

TEST (accelerateNotation, testRandomAVX){

  // Set up code for the computation we want to schedule. 

  Tensor<float> A("A", {16}, Format{Dense});
  Tensor<float> expected("expected", {16}, Format{Dense});
  Tensor<float> B("B", {16}, Format{Dense});
  Tensor<float> C("C", {16}, Format{Dense});
  IndexVar i("i");

  for (int i = 0; i < 16; i++) {
    C.insert({i}, (float) i);
    B.insert({i}, (float) i);
  }

  C.pack();
  B.pack();

  IndexExpr accelerateExpr = B(i) + C(i);
  A(i) = accelerateExpr;
  IndexStmt stmt = A.getAssignment().concretize();

  // Time how long it takes to schedule one time, we will delay to account for
  // the cost of computation while scheduling. 
  auto start1 = std::chrono::high_resolution_clock::now();
  IndexStmt newStmt = stmt.tile(new TileSaxpy(), accelerateExpr, {{i,  1}});
  auto end1 = std::chrono::high_resolution_clock::now();

  double count = 0;

  // for (int j = 0; j < 10; j++){
    auto start = std::chrono::high_resolution_clock::now();
    // Starting the random Search
    std::mt19937 rng(10);
    // uniform, unbiased. Bound by twice the capabilty of Stardust.
    std::uniform_int_distribution<int> gen(1, 65536*2); 

    int r1 = gen(rng);
    int j = 0;
    // Find at least one tiling that works.
    while (r1 > 4){
      // We delay to account for the time it takes to schedule the computation.
      // See up for timing logic.
      usleep(std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count());
      j++;
      r1 = gen(rng);
    }
    std::cout << r1 << std::endl;
    std::cout << j << std::endl;
    // Now start linear search to maximize tile size.
    while (r1 != 4){
      // We delay to account for the time it takes to schedule the computation.
      // See up for timing logic.
      usleep(std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count());
      r1++;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    count += duration.count();
  // }

  // Divide by 10 to average over 10 runs and then divide by 1000000
  // to convert microseconds into seconds.
  std::cout << "Time taken to Run Random Search for AVX: "
          << float(count)/float(1000000) << "s" << std::endl;

}


TEST (accelerateNotation, testLinearAVX){

  // Set up code for the computation we want to schedule. 
  Tensor<float> A("A", {16}, Format{Dense});
  Tensor<float> expected("expected", {16}, Format{Dense});
  Tensor<float> B("B", {16}, Format{Dense});
  Tensor<float> C("C", {16}, Format{Dense});
  IndexVar i("i");

  for (int i = 0; i < 16; i++) {
    C.insert({i}, (float) i);
    B.insert({i}, (float) i);
  }

  C.pack();
  B.pack();

  IndexExpr accelerateExpr = B(i) + C(i);
  A(i) = accelerateExpr;
  IndexStmt stmt = A.getAssignment().concretize();

  // Time how long it takes to schedule one time, we will delay to account for
  // the cost of computation while scheduling. 
  auto start1 = std::chrono::high_resolution_clock::now();
  IndexStmt newStmt = stmt.tile(new TileSaxpy(), accelerateExpr, {{i,  1}});
  auto end1 = std::chrono::high_resolution_clock::now();

  double count = 0;
  // for (int j = 0; j < 10; j++){
  auto start = std::chrono::high_resolution_clock::now();
  // Starting the random Search
  std::mt19937 rng(time(nullptr));
  // uniform, unbiased. Bound by twice the capabilty of Stardust.
  std::uniform_int_distribution<int> gen(1, 65536*2); 


  // Now start linear search.
  int r1 = 0;
  // Continue until to maximimum tile size is reached.
  while (r1<4){
    // We delay to account for the time it takes to schedule the computation.
    // See up for timing logic.
    usleep(std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count());
    r1++;
  }

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  count = duration.count();
  // Dvide by 1000000 to convert microseconds into seconds.
  std::cout << "Time taken to Run Linear Search for Stardust: "
          << float(count)/float(1000000) << "s" << std::endl;
}
