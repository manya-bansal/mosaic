#include <stdio.h>
#include <time.h>
#include "gemv-taco.c"

double run_gemv_taco(int dim){
    clock_t start, end;
    double cpu_time_used_ms;
    taco_tensor_t * A = init_taco_tensor_t(2, 0, (int32_t[]){dim, dim}, (int32_t[]){0, 0}, (taco_mode_t[]) {taco_mode_dense, taco_mode_dense});
    taco_tensor_t * b = init_taco_tensor_t(1, 0, (int32_t[]){dim}, (int32_t[]){0}, (taco_mode_t[]) {taco_mode_dense});
    taco_tensor_t * c = init_taco_tensor_t(1, 0, (int32_t[]){dim}, (int32_t[]){0}, (taco_mode_t[]) {taco_mode_dense});

    A->vals = malloc(sizeof(float)*dim*dim);
    b->vals = malloc(sizeof(float)*dim);
    c->vals = malloc(sizeof(float)*dim);

    float*  A_vals = (float*)(A->vals);
    float*  b_vals = (float*)(b->vals);
    float*  c_vals = (float*)(c->vals);

    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim; j++){
            A_vals[i*dim +j] = i+j;
        }
    }
    for(int i = 0; i < dim; i++){
        b_vals[i] = i;
    }
    for(int i = 0; i < dim; i++){
        c_vals[i] = i;
    }
    start = clock();
    compute(c, A, b);
    end = clock();
    cpu_time_used_ms = ((double) (end - start)) / (CLOCKS_PER_SEC/1000);
    return cpu_time_used_ms;
}

int main(int argc, char *argv[]) {
    for (int i = 100; i<=5000; i+=100){
        double time_taken = 0;
        for (int j = 0; j<10; j++){
            time_taken += run_gemv_taco(i);
        }
        printf("%d=%f\n", i, time_taken/10);
    }
    
}