#include <stdio.h>
#include <time.h>
#include "gemv-taco.c"

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

void print_array(float * vals, int num_elem){
    for (int i = 0; i < num_elem; i++){
    printf("elem[%d]=%f\n", i, vals[i]);
        }
        printf("next\n");
}

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
    deinit_taco_tensor_t(c);
    deinit_taco_tensor_t(A);
    deinit_taco_tensor_t(b);
    free(A_vals);
    free(c_vals);
    free(b_vals);
    return cpu_time_used_ms;
}

int main(int argc, char *argv[]) {
  for (int i = 100; i <= 5000; i += 100)
	{
		double time_taken[11];
		for (int j = 0; j < 11; j++)
		{
			time_taken[j] = run_gemv_taco(i);
		}
		array_sort(time_taken, 11);
		printf("%d=%f\n", i, time_taken[5]);
	}
}