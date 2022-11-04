#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gsl/gsl_vector.h"
#include "gsl/gsl_blas.h"


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

void set_gsl_float_data(gsl_vector_float * vec, float * data){
     vec->data = data;
}

void set_gsl_mat_data_row_major_s(gsl_matrix_float * mat, float * data){
     mat->data = data;
}
    
double run_gemv_gsl(int dim){
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

    gsl_vector_float *  var1;
    gsl_vector_float * var2;
    gsl_matrix_float * mat;
    var1 = gsl_vector_float_calloc(dim);
    var2 = gsl_vector_float_calloc(dim);
    set_gsl_float_data(var1, b_vals);
    set_gsl_float_data(var2, c_vals);
    mat = gsl_matrix_float_alloc(dim, dim);
    set_gsl_mat_data_row_major_s(mat, A_vals);
    start = clock();
    gsl_blas_sgemv(111, 1, mat, var1, 0, var2);
    end = clock();
    cpu_time_used_ms = ((double) (end - start)) / (CLOCKS_PER_SEC/1000);
    return cpu_time_used_ms;
}

int main(int argc, char *argv[]) {
    for (int i = 100; i<=5000; i+=100){
        double time_taken[11];
        for (int j = 0; j<11; j++){
            time_taken[j] = run_gemv_gsl(i);
        }
        array_sort(time_taken, 11);
        printf("%d=%f\n", i, time_taken[5]/10);
    }
}