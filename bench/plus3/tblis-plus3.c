#include <stdio.h>
#include <time.h>
#include "tblis-compute.c"

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

double run_gemv_tblis(int dim){
    clock_t start, end;
    double cpu_time_used_ms;
    taco_tensor_t * A = init_taco_tensor_t(3, 0, (int32_t[]){dim, dim, dim}, (int32_t[]){0, 0, 0}, (taco_mode_t[]) {taco_mode_dense, taco_mode_dense});
    taco_tensor_t * b = init_taco_tensor_t(3, 0, (int32_t[]){dim, dim, dim}, (int32_t[]){0, 0, 0}, (taco_mode_t[]) {taco_mode_dense});
    taco_tensor_t * c = init_taco_tensor_t(3, 0, (int32_t[]){dim, dim, dim}, (int32_t[]){0, 0, 0}, (taco_mode_t[]) {taco_mode_dense});

    A->vals = malloc(sizeof(float)*dim*dim*dim);
    b->vals = malloc(sizeof(float)*dim*dim*dim);
    c->vals = malloc(sizeof(float)*dim*dim*dim);

    float*  A_vals = (float*)(A->vals);
    float*  b_vals = (float*)(b->vals);
    float*  c_vals = (float*)(c->vals);

    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim; j++){
            for(int k = 0; k < dim; k++){
                A_vals[(i*dim +j)*dim + k] = i+j+k;
                b_vals[(i*dim +j)*dim + k] = i+j+k;
            }
        }
    }
    start = clock();
    compute(c, A, b);
    end = clock();
    cpu_time_used_ms = ((double) (end - start)) / (CLOCKS_PER_SEC/1000);
    return cpu_time_used_ms;
}

int main(int argc, char *argv[]) {
   for (int i = 10; i<=1000; i+=10){
        double time_taken[11];
        for (int j = 0; j<11; j++){
            time_taken[j] = run_gemv_tblis(i);
        }
        array_sort(time_taken, 11);
        printf("%d=%f\n", i, time_taken[5]/10);
    }
    
}