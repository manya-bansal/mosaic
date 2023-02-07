#ifndef TACO_C_HEADERS
#define TACO_C_HEADERS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <immintrin.h>
#include "tblis.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_blas.h"
#include "tensor.h"
#include "mkl.h"
#include <cuda_runtime_api.h>
#include <cusparse.h>
#if _OPENMP
#include <omp.h>
#endif
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)
#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED
typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;
typedef struct {
  int32_t      order;         // tensor order (number of modes)
  int32_t*     dimensions;    // tensor dimensions
  int32_t      csize;         // component size
  int32_t*     mode_ordering; // mode storage ordering
  taco_mode_t* mode_types;    // mode storage types
  uint8_t***   indices;       // tensor index data (per mode)
  uint8_t*     vals;          // tensor values
  uint8_t*     fill_value;    // tensor fill value
  int32_t      vals_size;     // values array size
} taco_tensor_t;
#endif
int cmp(const void *a, const void *b) {
  return *((const int*)a) - *((const int*)b);
}
int taco_gallop(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayStart] >= target || arrayStart >= arrayEnd) {
    return arrayStart;
  }
  int step = 1;
  int curr = arrayStart;
  while (curr + step < arrayEnd && array[curr + step] < target) {
    curr += step;
    step = step * 2;
  }

  step = step / 2;
  while (step > 0) {
    if (curr + step < arrayEnd && array[curr + step] < target) {
      curr += step;
    }
    step = step / 2;
  }
  return curr+1;
}
int taco_binarySearchAfter(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayStart] >= target) {
    return arrayStart;
  }
  int lowerBound = arrayStart; // always < target
  int upperBound = arrayEnd; // always >= target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return upperBound;
}
int taco_binarySearchBefore(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayEnd] <= target) {
    return arrayEnd;
  }
  int lowerBound = arrayStart; // always <= target
  int upperBound = arrayEnd; // always > target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return lowerBound;
}
taco_tensor_t* init_taco_tensor_t(int32_t order, int32_t csize,
                                  int32_t* dimensions, int32_t* mode_ordering,
                                  taco_mode_t* mode_types) {
  taco_tensor_t* t = (taco_tensor_t *) malloc(sizeof(taco_tensor_t));
  t->order         = order;
  t->dimensions    = (int32_t *) malloc(order * sizeof(int32_t));
  t->mode_ordering = (int32_t *) malloc(order * sizeof(int32_t));
  t->mode_types    = (taco_mode_t *) malloc(order * sizeof(taco_mode_t));
  t->indices       = (uint8_t ***) malloc(order * sizeof(uint8_t***));
  t->csize         = csize;
  for (int32_t i = 0; i < order; i++) {
    t->dimensions[i]    = dimensions[i];
    t->mode_ordering[i] = mode_ordering[i];
    t->mode_types[i]    = mode_types[i];
    switch (t->mode_types[i]) {
      case taco_mode_dense:
        t->indices[i] = (uint8_t **) malloc(1 * sizeof(uint8_t **));
        break;
      case taco_mode_sparse:
        t->indices[i] = (uint8_t **) malloc(2 * sizeof(uint8_t **));
        break;
    }
  }
  return t;
}
void deinit_taco_tensor_t(taco_tensor_t* t) {
  for (int i = 0; i < t->order; i++) {
    free(t->indices[i]);
  }
  free(t->indices);
  free(t->dimensions);
  free(t->mode_ordering);
  free(t->mode_types);
  free(t);
}
  float tblis_vector_dot_transfer(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_vector* A, const tblis_vector* B,
                      tblis_scalar* result){
   tblis_vector_dot(comm, cfg, A, B, result);
   return result->data.s; }

  void tblis_init_tensor_s_helper_row_major(tblis_tensor * t, int * dim, int num_dim, void * data){
    len_type * len = malloc(sizeof(len_type)*num_dim);
        if (!len){
          printf("error, len not valid!!!");
        }
    int stride_product = 1; 
    for (int i = 0; i < num_dim; i++){
        len[(len_type) i] = dim[i];
        stride_product *= dim[i];
    }
    stride_type * stride = malloc(sizeof(stride_type)*num_dim);
   for (int i = 0; i < num_dim; i++){
        stride_product /= dim[i];
        stride[(stride_type) i] = stride_product;
    }
    tblis_init_tensor_s(t, num_dim, len, data, stride);
}
  void tblis_set_vector(tblis_tensor * t, int * dim, int num_dim, void * data){
    len_type * len = malloc(sizeof(len_type)*num_dim);
    len[0] = dim[0];    stride_type * stride = malloc(sizeof(stride_type)*num_dim);
    stride[0] = 1;
    tblis_init_tensor_s(t, num_dim, len, data, stride);
}
 void set_gsl_float_data(gsl_vector_float * vec, float * data){
     vec->data = data;}
 void free_tblis_tensor(tblis_tensor * t){
     free(t->len);   free(t->stride);}
 void set_gsl_mat_data_row_major_s(gsl_matrix_float * mat, float * data){
     mat->data = data;}
 void set_tensor_data_s(tensor_float * t, float * data){
     t->data = data;}
 void print_array(float * vals, int num_elem){
      for (int i = 0; i < num_elem; i++){
      printf("elem[%d]=%f\n", i, vals[i]);
    }
    printf("next\n");
}
void sgemv_mkl_internal(int dim, float * A_vals, float * b_vals, float * d_vals){
    MKL_INT v2422 = dim;
    MKL_INT v2421 = 1;
    float zero = 0;
    float v2420 = 1;
    sgemv("n", &v2422, &v2422, &v2420, A_vals, &v2422, b_vals, &v2421, &zero, d_vals, &v2421);
}
void mkl_scsrgemv_internal(int m, taco_tensor_t * A, taco_tensor_t * b, taco_tensor_t * c)
{ sparse_matrix_t A_csr;
 struct matrix_descr desc;
 desc.type = SPARSE_MATRIX_TYPE_GENERAL;
 int*  A_pos = (int*)(A->indices[1][0]); mkl_sparse_s_create_csr(&A_csr, SPARSE_INDEX_BASE_ZERO, m, m, A_pos, A_pos+1, (int*)A->indices[1][1], (float*)A->vals);
 mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, (float)1, A_csr, desc, (float*)b->vals, (float)0, (float*)c->vals);
}
void mkl_sparse_s_mm_internal(int m, taco_tensor_t * A, taco_tensor_t * b, float * c)
{ sparse_matrix_t A_csr;
 struct matrix_descr desc;
 desc.type = SPARSE_MATRIX_TYPE_GENERAL;
 int*  A_pos = (int*)(A->indices[1][0]); mkl_sparse_s_create_csr(&A_csr, SPARSE_INDEX_BASE_ZERO, m, m, A_pos, A_pos+1, (int*)A->indices[1][1], (float*)A->vals);
 mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, (float)1, A_csr, desc, SPARSE_LAYOUT_ROW_MAJOR, (float*)b->vals, m, m, 0, c, m);
}
#include "cuda-wrappers.h"
#include "mkl-wrappers.h"
int lC = 0;
int kB = 0;
#endif

int assemble(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C1, taco_tensor_t *C2) {

  int A1_dimension = (int)(A->dimensions[0]);
  int A2_dimension = (int)(A->dimensions[1]);
  int A3_dimension = (int)(A->dimensions[2]);
  int A4_dimension = (int)(A->dimensions[3]);
  float*  A_vals = (float*)(A->vals);

  A_vals = (float*)malloc(sizeof(float) * (((A1_dimension * A2_dimension) * A3_dimension) * A4_dimension));

  A->vals = (uint8_t*)A_vals;
  return 0;
}

int compute(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C1, taco_tensor_t *C2, taco_tensor_t *C3, taco_tensor_t *C4) {

  int A1_dimension = (int)(A->dimensions[0]);
  int A2_dimension = (int)(A->dimensions[1]);
  int A3_dimension = (int)(A->dimensions[2]);
  int A4_dimension = (int)(A->dimensions[3]);
  float*  A_vals = (float*)(A->vals);
  float A_fill_value = *((float*)(A->fill_value));
  int B2_dimension = (int)(B->dimensions[1]);
  int B4_dimension = (int)(B->dimensions[3]);
  int*  B1_pos = (int*)(B->indices[0][0]);
  int*  B1_crd = (int*)(B->indices[0][1]);
  int*  B3_pos = (int*)(B->indices[2][0]);
  int*  B3_crd = (int*)(B->indices[2][1]);
  float*  B_vals = (float*)(B->vals);
  float*  C1_vals = (float*)(C1->vals);
  float*  C2_vals = (float*)(C2->vals);
  int C11_dimension = (int)(C1->dimension[0]);

  #pragma omp parallel for schedule(static)
  for (int32_t pA = 0; pA < (((A1_dimension * A2_dimension) * A3_dimension) * A4_dimension); pA++) {
    A_vals[pA] = A_fill_value;
  }

  float* restrict A2373 = 0;
  A2373 = (float*)calloc(sizeof(float) * C11_dimension * C11_dimension);
tblis_tensor var1;
tblis_tensor var2;
tblis_tensor result;
tblis_init_tensor_s_helper_row_major(&var1, C2->dimensions, 2, C1_vals);
tblis_init_tensor_s_helper_row_major(&var2, C2->dimensions, 2, C2_vals);
tblis_init_tensor_s_helper_row_major(&result, C2->dimensions, 2, A2373);
tblis_tensor_mult(NULL, NULL, &var1, "ij", &var2, "jk", &result, "ik");
free_tblis_tensor(&var1);
free_tblis_tensor(&var2);
free_tblis_tensor(&result);

  float* restrict A2374 = 0;
  A2374 = (float*)calloc(sizeof(float) * C11_dimension * C11_dimension);

tblis_tensor var1;
tblis_tensor var2;
tblis_tensor result;
tblis_init_tensor_s_helper_row_major(&var1, C2->dimensions, 2, C1_vals);
tblis_init_tensor_s_helper_row_major(&var2, C2->dimensions, 2, C2_vals);
tblis_init_tensor_s_helper_row_major(&result, C2->dimensions, 2, A2373);
tblis_tensor_mult(NULL, NULL, &var1, "ij", &var2, "jk", &result, "ik");
free_tblis_tensor(&var1);
free_tblis_tensor(&var2);
free_tblis_tensor(&result);

  for (int32_t iB = B1_pos[0]; iB < B1_pos[1]; iB++) {
    int32_t i = B1_crd[iB];
    for (int32_t j = 0; j < B2_dimension; j++) {
      int32_t jA = i * A2_dimension + j;
      int32_t jB = iB * B2_dimension + j;
      int32_t jA2373 = i * C11_dimension + j;
      for (int32_t kB = B3_pos[jB]; kB < B3_pos[(jB + 1)]; kB++) {
        int32_t k = B3_crd[kB];
        int32_t kA = jA * A3_dimension + k;
        for (int32_t l = 0; l < B4_dimension; l++) {
          int32_t lA = kA * A4_dimension + l;
          int32_t lB = kB * B4_dimension + l;
          int32_t jA2374 = k * C11_dimension + l;
          A_vals[lA] = A_vals[lA] + B_vals[lB] * A2373[jA2373] * A2374[jA2374];
        }
      }
    }
  }
  free(A2373);

  A->vals = (uint8_t*)A_vals;
  return 0;
}
#include "/home/ubuntu/tmp/taco_tmp_MIaZVF/6geqatrds9yv.h"
int _shim_assemble(void** parameterPack) {
  return assemble((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]), (taco_tensor_t*)(parameterPack[3]));
}
int _shim_compute(void** parameterPack) {
  return compute((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]), (taco_tensor_t*)(parameterPack[3]));
}
