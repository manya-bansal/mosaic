#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <algorithm>
#include <unordered_set>
#include <taco.h>

#include "taco/ir/ir_visitor.h"
#include "codegen_c.h"
#include "taco/error.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"

using namespace std;

namespace taco {
namespace ir {

// Some helper functions
namespace {

// Include stdio.h for printf
// stdlib.h for malloc/realloc
// math.h for sqrt
// MIN preprocessor macro
// This *must* be kept in sync with taco_tensor_t.h
const string cHeaders =
  "#ifndef TACO_C_HEADERS\n"
  "#define TACO_C_HEADERS\n"
  "#include <stdio.h>\n"
  "#include <stdlib.h>\n"
  "#include <stdint.h>\n"
  "#include <stdbool.h>\n"
  "#include <math.h>\n"
  "#include <complex.h>\n"
  "#include <string.h>\n"
  "#include <immintrin.h>\n"
  // "#include \"cblas.h\"\n"
  "#include \"tblis.h\"\n"
  "#include \"gsl/gsl_vector.h\"\n"
  "#include \"gsl/gsl_blas.h\"\n"
  "#include \"tensor.h\"\n"
  "#include \"mkl.h\"\n"
  "#include <cuda_runtime_api.h>\n"
  "#include <cusparse.h>\n"
  "#include <time.h>\n"
 
  // "#include <Accelerate/Accelerate.h>\n"
  "#if _OPENMP\n"
  "#include <omp.h>\n"
  "#endif\n"
  "#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))\n"
  "#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))\n"
  "#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)\n"
  "#ifndef TACO_TENSOR_T_DEFINED\n"
  "#define TACO_TENSOR_T_DEFINED\n"
  "typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;\n"
  "typedef struct {\n"
  "  int32_t      order;         // tensor order (number of modes)\n"
  "  int32_t*     dimensions;    // tensor dimensions\n"
  "  int32_t      csize;         // component size\n"
  "  int32_t*     mode_ordering; // mode storage ordering\n"
  "  taco_mode_t* mode_types;    // mode storage types\n"
  "  uint8_t***   indices;       // tensor index data (per mode)\n"
  "  uint8_t*     vals;          // tensor values\n"
  "  uint8_t*     fill_value;    // tensor fill value\n"
  "  int32_t      vals_size;     // values array size\n"
  "} taco_tensor_t;\n"
  "#endif\n"
  // "#if !_OPENMP\n"
  // "int omp_get_thread_num() { return 0; }\n"
  // "int omp_get_max_threads() { return 1; }\n"
  // "#endif\n"
  "int cmp(const void *a, const void *b) {\n"
  "  return *((const int*)a) - *((const int*)b);\n"
  "}\n"
  // Increment arrayStart until array[arrayStart] >= target or arrayStart >= arrayEnd
  // using an exponential search algorithm: https://en.wikipedia.org/wiki/Exponential_search.
  "int taco_gallop(int *array, int arrayStart, int arrayEnd, int target) {\n"
  "  if (array[arrayStart] >= target || arrayStart >= arrayEnd) {\n"
  "    return arrayStart;\n"
  "  }\n"
  "  int step = 1;\n"
  "  int curr = arrayStart;\n"
  "  while (curr + step < arrayEnd && array[curr + step] < target) {\n"
  "    curr += step;\n"
  "    step = step * 2;\n"
  "  }\n"
  "\n"
  "  step = step / 2;\n"
  "  while (step > 0) {\n"
  "    if (curr + step < arrayEnd && array[curr + step] < target) {\n"
  "      curr += step;\n"
  "    }\n"
  "    step = step / 2;\n"
  "  }\n"
  "  return curr+1;\n"
  "}\n"
  "int taco_binarySearchAfter(int *array, int arrayStart, int arrayEnd, int target) {\n"
  "  if (array[arrayStart] >= target) {\n"
  "    return arrayStart;\n"
  "  }\n"
  "  int lowerBound = arrayStart; // always < target\n"
  "  int upperBound = arrayEnd; // always >= target\n"
  "  while (upperBound - lowerBound > 1) {\n"
  "    int mid = (upperBound + lowerBound) / 2;\n"
  "    int midValue = array[mid];\n"
  "    if (midValue < target) {\n"
  "      lowerBound = mid;\n"
  "    }\n"
  "    else if (midValue > target) {\n"
  "      upperBound = mid;\n"
  "    }\n"
  "    else {\n"
  "      return mid;\n"
  "    }\n"
  "  }\n"
  "  return upperBound;\n"
  "}\n"
  "int taco_binarySearchBefore(int *array, int arrayStart, int arrayEnd, int target) {\n"
  "  if (array[arrayEnd] <= target) {\n"
  "    return arrayEnd;\n"
  "  }\n"
  "  int lowerBound = arrayStart; // always <= target\n"
  "  int upperBound = arrayEnd; // always > target\n"
  "  while (upperBound - lowerBound > 1) {\n"
  "    int mid = (upperBound + lowerBound) / 2;\n"
  "    int midValue = array[mid];\n"
  "    if (midValue < target) {\n"
  "      lowerBound = mid;\n"
  "    }\n"
  "    else if (midValue > target) {\n"
  "      upperBound = mid;\n"
  "    }\n"
  "    else {\n"
  "      return mid;\n"
  "    }\n"
  "  }\n"
  "  return lowerBound;\n"
  "}\n"
  "taco_tensor_t* init_taco_tensor_t(int32_t order, int32_t csize,\n"
  "                                  int32_t* dimensions, int32_t* mode_ordering,\n"
  "                                  taco_mode_t* mode_types) {\n"
  "  taco_tensor_t* t = (taco_tensor_t *) malloc(sizeof(taco_tensor_t));\n"
  "  t->order         = order;\n"
  "  t->dimensions    = (int32_t *) malloc(order * sizeof(int32_t));\n"
  "  t->mode_ordering = (int32_t *) malloc(order * sizeof(int32_t));\n"
  "  t->mode_types    = (taco_mode_t *) malloc(order * sizeof(taco_mode_t));\n"
  "  t->indices       = (uint8_t ***) malloc(order * sizeof(uint8_t***));\n"
  "  t->csize         = csize;\n"
  "  for (int32_t i = 0; i < order; i++) {\n"
  "    t->dimensions[i]    = dimensions[i];\n"
  "    t->mode_ordering[i] = mode_ordering[i];\n"
  "    t->mode_types[i]    = mode_types[i];\n"
  "    switch (t->mode_types[i]) {\n"
  "      case taco_mode_dense:\n"
  "        t->indices[i] = (uint8_t **) malloc(1 * sizeof(uint8_t **));\n"
  "        break;\n"
  "      case taco_mode_sparse:\n"
  "        t->indices[i] = (uint8_t **) malloc(2 * sizeof(uint8_t **));\n"
  "        break;\n"
  "    }\n"
  "  }\n"
  "  return t;\n"
  "}\n"
  "void deinit_taco_tensor_t(taco_tensor_t* t) {\n"
  "  for (int i = 0; i < t->order; i++) {\n"
  "    free(t->indices[i]);\n"
  "  }\n"
  "  free(t->indices);\n"
  "  free(t->dimensions);\n"
  "  free(t->mode_ordering);\n"
  "  free(t->mode_types);\n"
  "  free(t);\n"
  "}\n"
  "  float tblis_vector_dot_transfer(const tblis_comm* comm, const tblis_config* cfg,\n"
  "                      const tblis_vector* A, const tblis_vector* B,\n"
  "                      tblis_scalar* result){\n"
  "   tblis_vector_dot(comm, cfg, A, B, result);\n"
  "   return result->data.s; }\n\n"
  "  void tblis_init_tensor_s_helper_row_major(tblis_tensor * t, int * dim, int num_dim, void * data){\n"
  "    len_type * len = malloc(sizeof(len_type)*num_dim);\n"
  "        if (!len){\n" 
  "          printf(\"error, len not valid!!!\");\n"
  "        }\n"
  "    int stride_product = 1; \n"
  "    for (int i = 0; i < num_dim; i++){\n"
  "        len[(len_type) i] = dim[i];\n"
  "        stride_product *= dim[i];\n"
  "    }\n"
  "    stride_type * stride = malloc(sizeof(stride_type)*num_dim);\n"
  // "        if (!stride){\n" 
  // "          printf(\"error, stride not valid!!!\");\n"
  // "        }\n"
  "   for (int i = 0; i < num_dim; i++){\n"
  "        stride_product /= dim[i];\n"
  "        stride[(stride_type) i] = stride_product;\n"
  "    }\n"
  "    tblis_init_tensor_s(t, num_dim, len, data, stride);\n"
  "}\n"
  "  void tblis_set_vector(tblis_tensor * t, int * dim, int num_dim, void * data){\n"
  "    len_type * len = malloc(sizeof(len_type)*num_dim);\n"
  "    len[0] = dim[0];"
  "    stride_type * stride = malloc(sizeof(stride_type)*num_dim);\n"
  "    stride[0] = 1;\n"
  "    tblis_init_tensor_s(t, num_dim, len, data, stride);\n"
  "}\n"
  " void set_gsl_float_data(gsl_vector_float * vec, float * data){\n  "
  "   vec->data = data;"
  "}\n"
  " void free_tblis_tensor(tblis_tensor * t){\n  "
  "   free(t->len);"
  "   free(t->stride);"
  "}\n"
  " void set_gsl_mat_data_row_major_s(gsl_matrix_float * mat, float * data){\n  "
  "   mat->data = data;"
  "}\n"
  " void set_tensor_data_s(tensor_float * t, float * data){\n  "
  "   t->data = data;"
  "}\n"
  " void print_array(float * vals, int num_elem){\n  "
  "    for (int i = 0; i < num_elem; i++){\n"
  "      printf(\"elem[%d]=%f\\n\", i, vals[i]);\n"
  "    }\n"
  "    printf(\"next\\n\");\n"
  "}\n"
  "void sgemv_mkl_internal(int dim, float * A_vals, float * b_vals, float * d_vals){\n"
  "    MKL_INT v2422 = dim;\n"
  "    MKL_INT v2421 = 1;\n"
  "    float zero = 0;\n"
  "    float v2420 = 1;\n"
  "    sgemv(\"n\", &v2422, &v2422, &v2420, A_vals, &v2422, b_vals, &v2421, &zero, d_vals, &v2421);\n"
  "}\n"
  "void mkl_scsrgemv_internal(int m, taco_tensor_t * A, taco_tensor_t * b, taco_tensor_t * c)\n{"
  " sparse_matrix_t A_csr;\n"
  " struct matrix_descr desc;\n"
  " desc.type = SPARSE_MATRIX_TYPE_GENERAL;\n"
  " int*  A_pos = (int*)(A->indices[1][0]);"
  " mkl_sparse_s_create_csr(&A_csr, SPARSE_INDEX_BASE_ZERO, m, m, A_pos, A_pos+1, (int*)A->indices[1][1], (float*)A->vals);\n"
  " mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, (float)1, A_csr, desc, (float*)b->vals, (float)0, (float*)c->vals);\n"
  "}\n"
  "void mkl_sparse_s_mm_internal(int m, taco_tensor_t * A, taco_tensor_t * b, float * c)\n{"
  " sparse_matrix_t A_csr;\n"
  " struct matrix_descr desc;\n"
  " desc.type = SPARSE_MATRIX_TYPE_GENERAL;\n"
  " int*  A_pos = (int*)(A->indices[1][0]);"
  " mkl_sparse_s_create_csr(&A_csr, SPARSE_INDEX_BASE_ZERO, m, m, A_pos, A_pos+1, (int*)A->indices[1][1], (float*)A->vals);\n"
  " mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, (float)1, A_csr, desc, SPARSE_LAYOUT_ROW_MAJOR, (float*)b->vals, m, m, 0, c, m);\n"
  "}\n"
    "#include \"cuda-wrappers.h\"\n"
    "#include \"mkl-wrappers.h\"\n"
  "int lC = 0;\n"
  "int kB = 0;\n"
  "#endif\n";
} // anonymous namespace

// find variables for generating declarations
// generates a single var for each GetProperty
class CodeGen_C::FindVars : public IRVisitor {
public:
  map<Expr, string, ExprCompare> varMap;

  // the variables for which we need to add declarations
  map<Expr, string, ExprCompare> varDecls;

  vector<Expr> localVars;

  // this maps from tensor, property, mode, index to the unique var
  map<tuple<Expr, TensorProperty, int, int>, string> canonicalPropertyVar;

  // this is for convenience, recording just the properties unpacked
  // from the output tensor so we can re-save them at the end
  map<tuple<Expr, TensorProperty, int, int>, string> outputProperties;

  // TODO: should replace this with an unordered set
  vector<Expr> outputTensors;
  vector<Expr> inputTensors;

  CodeGen_C *codeGen;

  // copy inputs and outputs into the map
  FindVars(vector<Expr> inputs, vector<Expr> outputs, CodeGen_C *codeGen)
  : codeGen(codeGen) {
    for (auto v: inputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Inputs must be vars in codegen";
      taco_iassert(varMap.count(var)==0) << "Duplicate input found in codegen";
      inputTensors.push_back(v);
      varMap[var] = var->name;
    }
    for (auto v: outputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Outputs must be vars in codegen";
      taco_iassert(varMap.count(var)==0) << "Duplicate output found in codegen";
      outputTensors.push_back(v);
      varMap[var] = var->name;
    }
  }

protected:
  using IRVisitor::visit;

  virtual void visit(const Var *op) {
    if (varMap.count(op) == 0) {
      varMap[op] = op->is_ptr? op->name : codeGen->genUniqueName(op->name);
    }
  }

  virtual void visit(const VarDecl *op) {
    if (!util::contains(localVars, op->var)) {
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->rhs.accept(this);
  }

  virtual void visit(const For *op) {
    if (!util::contains(localVars, op->var)) {
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->start.accept(this);
    op->end.accept(this);
    op->increment.accept(this);
    op->contents.accept(this);
  }

  bool checkMatchVar(vector<Expr> exprs, Expr expr){
    if (!isa<Var>(expr)){
      return false;
    }
    const ir::Var * exprVar = to<Var>(expr);

    for (auto e: exprs){
      if (isa<Var>(e)) {
        const ir::Var * eVar = to<Var>(e);
        if (eVar->name == exprVar->name){
          return true;
        }
      }
    }
    return false;
  }

  virtual void visit(const GetProperty *op) {
    if (!checkMatchVar(inputTensors, op->tensor) &&
        !checkMatchVar(outputTensors, op->tensor)) {
      // Don't create header unpacking code for temporaries
      return;
    }

    if (varMap.count(op) == 0) {
      auto key =
              tuple<Expr,TensorProperty,int,int>(op->tensor,op->property,
                                                 (size_t)op->mode,
                                                 (size_t)op->index);
      if (canonicalPropertyVar.count(key) > 0) {
        varMap[op] = canonicalPropertyVar[key];
      } else {
        auto unique_name = codeGen->genUniqueName(op->name);
        canonicalPropertyVar[key] = unique_name;
        varMap[op] = unique_name;
        varDecls[op] = unique_name;
        if (util::contains(outputTensors, op->tensor)) {
          outputProperties[key] = unique_name;
        }
      }
    }
  }
};

CodeGen_C::CodeGen_C(std::ostream &dest, OutputKind outputKind, bool simplify)
    : CodeGen(dest, false, simplify, C), out(dest), outputKind(outputKind) {}

CodeGen_C::~CodeGen_C() {}

void CodeGen_C::compile(Stmt stmt, bool isFirst) {
  varMap = {};
  localVars = {};

  if (isFirst) {
    // output the headers
    out << cHeaders;
  }
  out << endl;
  // generate code for the Stmt
  stmt.accept(this);
}

void CodeGen_C::visit(const Function* func) {
  // if generating a header, protect the function declaration with a guard
  if (outputKind == HeaderGen) {
    out << "#ifndef TACO_GENERATED_" << func->name << "\n";
    out << "#define TACO_GENERATED_" << func->name << "\n";
  }

  int numYields = countYields(func);
  emittingCoroutine = (numYields > 0);
  funcName = func->name;
  labelCount = 0;

  resetUniqueNameCounters();
  FindVars inputVarFinder(func->inputs, {}, this);
  func->body.accept(&inputVarFinder);
  FindVars outputVarFinder({}, func->outputs, this);
  func->body.accept(&outputVarFinder);

  // output function declaration
  doIndent();
  out << printFuncName(func, inputVarFinder.varDecls, outputVarFinder.varDecls);

  // if we're just generating a header, this is all we need to do
  if (outputKind == HeaderGen) {
    out << ";\n";
    out << "#endif\n";
    return;
  }

  out << " {\n";

  indent++;

  // find all the vars that are not inputs or outputs and declare them
  resetUniqueNameCounters();
  FindVars varFinder(func->inputs, func->outputs, this);
  // out << "//" << util::join(func->inputs) << endl;
  func->body.accept(&varFinder);
  // out << "//";
  // for (auto& vardecl: varFinder.varDecls){
  //   out << vardecl.first << "->" << vardecl.second << "|";
  // }
  // out << "varmap ";
  // cout << "####" << func->body << "####" << endl;
  // for (auto& vardecl: varFinder.varMap){
  //   out << vardecl.first << "->" << vardecl.second << "|";
  // }
  out << endl;
  varMap = varFinder.varMap;
  localVars = varFinder.localVars;

  // Print variable declarations
  out << printDecls(varFinder.varDecls, func->inputs, func->outputs) << endl;

  if (emittingCoroutine) {
    out << printContextDeclAndInit(varMap, localVars, numYields, func->name)
        << endl;
  }

  // output body
  print(func->body);

  // output repack only if we allocated memory
  if (checkForAlloc(func))
    out << endl << printPack(varFinder.outputProperties, func->outputs);

  if (emittingCoroutine) {
    out << printCoroutineFinish(numYields, funcName);
  }

  doIndent();
  out << "return 0;\n";
  indent--;

  doIndent();
  out << "}\n";
}

void CodeGen_C::visit(const VarDecl* op) {
  if (emittingCoroutine) {
    doIndent();
    op->var.accept(this);
    parentPrecedence = Precedence::TOP;
    stream << " = ";
    op->rhs.accept(this);
    stream << ";";
    stream << endl;
  } else {
    IRPrinter::visit(op);
  }
}

void CodeGen_C::visit(const Yield* op) {
  printYield(op, localVars, varMap, labelCount, funcName);
}

// For Vars, we replace their names with the generated name,
// since we match by reference (not name)
void CodeGen_C::visit(const Var* op) {
  taco_iassert(varMap.count(op) > 0) <<
      "Var " << op->name << " not found in varMap";
  if (emittingCoroutine) {
//    out << "TACO_DEREF(";
  }
  out << varMap[op];
  if (emittingCoroutine) {
//    out << ")";
  }
}

static string genVectorizePragma(int width) {
  stringstream ret;
  ret << "#pragma clang loop interleave(enable) ";
  if (!width)
    ret << "vectorize(enable)";
  else
    ret << "vectorize_width(" << width << ")";

  return ret.str();
}

static string getParallelizePragma(LoopKind kind) {
  stringstream ret;
  ret << "#pragma omp parallel for schedule";
  switch (kind) {
    case LoopKind::Static:
      ret << "(static, 1)";
      break;
    case LoopKind::Dynamic:
      ret << "(dynamic, 1)";
      break;
    case LoopKind::Runtime:
      ret << "(runtime)";
      break;
    case LoopKind::Static_Chunked:
      ret << "(static)";
      break;
    default:
      break;
  }
  return ret.str();
}

static string getUnrollPragma(size_t unrollFactor) {
  return "#pragma unroll " + std::to_string(unrollFactor);
}

static string getAtomicPragma() {
  return "#pragma omp atomic";
}

// The next two need to output the correct pragmas depending
// on the loop kind (Serial, Static, Dynamic, Vectorized)
//
// Docs for vectorization pragmas:
// http://clang.llvm.org/docs/LanguageExtensions.html#extensions-for-loop-hint-optimizations
void CodeGen_C::visit(const For* op) {
  switch (op->kind) {
    case LoopKind::Vectorized:
      doIndent();
      out << genVectorizePragma(op->vec_width);
      out << "\n";
      break;
    case LoopKind::Static:
    case LoopKind::Dynamic:
    case LoopKind::Runtime:
    case LoopKind::Static_Chunked:
      doIndent();
      out << getParallelizePragma(op->kind);
      out << "\n";
      break;
    default:
      if (op->unrollFactor > 0) {
        doIndent();
        out << getUnrollPragma(op->unrollFactor) << endl;
      }
      break;
  }

  doIndent();
  stream << keywordString("for") << " (";
  if (!emittingCoroutine) {
    stream << keywordString(util::toString(op->var.type())) << " ";
  }
  op->var.accept(this);
  stream << " = ";
  op->start.accept(this);
  stream << keywordString("; ");
  op->var.accept(this);
  stream << " < ";
  parentPrecedence = BOTTOM;
  op->end.accept(this);
  stream << keywordString("; ");
  op->var.accept(this);

  auto lit = op->increment.as<Literal>();
  if (lit != nullptr && ((lit->type.isInt()  && lit->equalsScalar(1)) ||
                         (lit->type.isUInt() && lit->equalsScalar(1)))) {
    stream << "++";
  }
  else {
    stream << " += ";
    op->increment.accept(this);
  }
  stream << ") {\n";

  op->contents.accept(this);
  doIndent();
  stream << "}";
  stream << endl;
}

void CodeGen_C::visit(const While* op) {
  // it's not clear from documentation that clang will vectorize
  // while loops
  // however, we'll output the pragmas anyway
  if (op->kind == LoopKind::Vectorized) {
    doIndent();
    out << genVectorizePragma(op->vec_width);
    out << "\n";
  }

  IRPrinter::visit(op);
}

void CodeGen_C::visit(const GetProperty* op) {
  taco_iassert(varMap.count(op) > 0) <<
      "Property " << Expr(op) << " of " << op->tensor << " not found in varMap";
  out << varMap[op];
}

void CodeGen_C::visit(const Min* op) {
  if (op->operands.size() == 1) {
    op->operands[0].accept(this);
    return;
  }
  const auto opString = op->type.isFloat() ? "fmin" : "TACO_MIN";
  for (size_t i=0; i<op->operands.size()-1; i++) {
    stream << opString << "(";
    op->operands[i].accept(this);
    stream << ",";
  }
  op->operands.back().accept(this);
  for (size_t i=0; i<op->operands.size()-1; i++) {
    stream << ")";
  }
}

void CodeGen_C::visit(const Max* op) {
  if (op->operands.size() == 1) {
    op->operands[0].accept(this);
    return;
  }
  for (size_t i=0; i<op->operands.size()-1; i++) {
    stream << "TACO_MAX(";
    op->operands[i].accept(this);
    stream << ",";
  }
  op->operands.back().accept(this);
  for (size_t i=0; i<op->operands.size()-1; i++) {
    stream << ")";
  }
}

void CodeGen_C::visit(const Allocate* op) {
  string elementType = printCType(op->var.type(), false);

  doIndent();
  op->var.accept(this);
  stream << " = (";
  stream << elementType << "*";
  stream << ")";
  if (op->is_realloc) {
    stream << "realloc(";
    op->var.accept(this);
    stream << ", ";
  }
  else {
    // If the allocation was requested to clear the allocated memory,
    // use calloc instead of malloc.
    if (op->clear) {
      stream << "calloc(1, ";
    } else {
      stream << "malloc(";
    }
  }
  stream << "sizeof(" << elementType << ")";
  stream << " * ";
  parentPrecedence = MUL;
  op->num_elements.accept(this);
  parentPrecedence = TOP;
  stream << ");";
    stream << endl;
}

void CodeGen_C::visit(const Sqrt* op) {
  taco_tassert(op->type.isFloat() && op->type.getNumBits() == 64) <<
      "Codegen doesn't currently support non-double sqrt";
  stream << "sqrt(";
  op->a.accept(this);
  stream << ")";
}

void CodeGen_C::visit(const Assign* op) {
  if (op->use_atomics) {
    doIndent();
    stream << getAtomicPragma() << endl;
  }
  IRPrinter::visit(op);
}

void CodeGen_C::visit(const Store* op) {
  if (op->use_atomics) {
    doIndent();
    stream << getAtomicPragma() << endl;
  }
  IRPrinter::visit(op);
}

void CodeGen_C::generateShim(const Stmt& func, stringstream &ret) {
  const Function *funcPtr = func.as<Function>();

  ret << "int _shim_" << funcPtr->name << "(void** parameterPack) {\n";
  ret << "  return " << funcPtr->name << "(";

  size_t i=0;
  string delimiter = "";

  const auto returnType = funcPtr->getReturnType();
  if (returnType.second != Datatype()) {
    ret << "(void**)(parameterPack[0]), ";
    ret << "(char*)(parameterPack[1]), ";
    ret << "(" << returnType.second << "*)(parameterPack[2]), ";
    ret << "(int32_t*)(parameterPack[3])";

    i = 4;
    delimiter = ", ";
  }

  for (auto output : funcPtr->outputs) {
    auto var = output.as<Var>();
    auto cast_type = var->is_tensor ? "taco_tensor_t*"
    : printCType(var->type, var->is_ptr);

    ret << delimiter << "(" << cast_type << ")(parameterPack[" << i++ << "])";
    delimiter = ", ";
  }
  for (auto input : funcPtr->inputs) {
    auto var = input.as<Var>();
    auto cast_type = var->is_tensor ? "taco_tensor_t*"
    : printCType(var->type, var->is_ptr);
    ret << delimiter << "(" << cast_type << ")(parameterPack[" << i++ << "])";
    delimiter = ", ";
  }
  ret << ");\n";
  ret << "}\n";
}
}
}
