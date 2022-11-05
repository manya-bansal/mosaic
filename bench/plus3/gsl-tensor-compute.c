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
#include "gsl/gsl_vector.h"
#include "gsl/gsl_blas.h"
#include "tensor.h"
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
#if !_OPENMP
int omp_get_thread_num() { return 0; }
int omp_get_max_threads() { return 1; }
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
 void set_gsl_float_data(gsl_vector_float * vec, float * data){
     vec->data = data;}
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
#endif

int assemble(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C) {

  int A1_dimension = (int)(A->dimensions[0]);
  int A2_dimension = (int)(A->dimensions[1]);
  int A3_dimension = (int)(A->dimensions[2]);
  float*  A_vals = (float*)(A->vals);

  A_vals = (float*)malloc(sizeof(float) * ((A1_dimension * A2_dimension) * A3_dimension));

  A_vals = (float*)malloc(sizeof(float) * ((A1_dimension * A2_dimension) * A3_dimension));

  A->vals = (uint8_t*)A_vals;
  return 0;
}

int compute(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C) {

  int A1_dimension = (int)(A->dimensions[0]);
  int A2_dimension = (int)(A->dimensions[1]);
  int A3_dimension = (int)(A->dimensions[2]);
  float*  A_vals = (float*)(A->vals);
  int B1_dimension = (int)(B->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  int B3_dimension = (int)(B->dimensions[2]);
  float*  B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float*  C_vals = (float*)(C->vals);

  for (int32_t i2433 = 0; i2433 < B1_dimension; i2433++) {
    for (int32_t i2434 = 0; i2434 < B2_dimension; i2434++) {
      int32_t i2434A = i2433 * A2_dimension + i2434;
      int32_t i2434B = i2433 * B2_dimension + i2434;
      for (int32_t i2435 = 0; i2435 < B3_dimension; i2435++) {
        int32_t i2435A = i2434A * A3_dimension + i2435;
        int32_t i2435B = i2434B * B3_dimension + i2435;
        A_vals[i2435A] = B_vals[i2435B];
      }
    }
  }
tensor_float * var1;
tensor_float * var2;
  var1 = tensor_float_alloc(3, C1_dimension);
  var2 = tensor_float_alloc(3, C1_dimension);
set_tensor_data_s(var1, C_vals);
set_tensor_data_s(var2, A_vals);
tensor_float_add(var2, var1);
  return 0;
}
#include "/home/manya227/temp/taco_tmp_7w1qFd/08knenw0t4as.h"
int _shim_assemble(void** parameterPack) {
  return assemble((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]));
}
int _shim_compute(void** parameterPack) {
  return compute((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]));
}
