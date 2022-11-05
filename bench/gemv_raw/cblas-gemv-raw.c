#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cblas.h"

// from https://www.includehelp.com/c-programs/calculate-median-of-an-array.aspx#:~:text=To%20calculate%20the%20median%20first,be%20considered%20as%20the%20median.
void array_sort(double *array , int n)
{ 
    // declare some local variables
    int i=0 , j=0 ;
    double temp=0;

    for(i=0 ; i<n ; i++)
    {
        for(j=0 ; j<n-1 ; j++)
        {
            if(array[j]>array[j+1])
            {
                temp        = array[j];
                array[j]    = array[j+1];
                array[j+1]  = temp;
            }
        }
    }

}

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
        double time_taken[11];
        for (int j = 0; j<11; j++){
            time_taken[j] = run_gemv_cblas(i);
        }
        array_sort(time_taken, 11);
        printf("%d=%f\n", i, time_taken[5]);
    }
}