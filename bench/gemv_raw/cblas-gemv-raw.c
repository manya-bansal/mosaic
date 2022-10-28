#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cblas.h"

double run_gemv_cblas(int dim){

    clock_t start, end;
    double cpu_time_used_ms;
    float*  A_vals = malloc(sizeof(float)*dim*dim);
    float*  b_vals = malloc(sizeof(float)*dim);
    float*  c_vals = malloc(sizeof(float)*dim);

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
    cblas_sgemv(CblasRowMajor, CblasNoTrans, dim, dim, 1, A_vals, dim, b_vals, 1, 0, c_vals, 1);
    end = clock();
    cpu_time_used_ms = ((double) (end - start)) / (CLOCKS_PER_SEC/1000);
    return cpu_time_used_ms;
}

int main(int argc, char *argv[]) {
    for (int i = 100; i<=5000; i+=100){
        double time_taken = 0;
        for (int j = 0; j<10; j++){
            time_taken += run_gemv_cblas(i);
        }
        printf("%d=%f\n", i, time_taken/10);
    }
}