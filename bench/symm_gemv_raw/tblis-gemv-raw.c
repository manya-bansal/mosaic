#include <stdio.h>
#include <time.h>
#include "tblis/tblis.h"

// from https://www.includehelp.com/c-programs/calculate-median-of-an-array.aspx#:~:text=To%20calculate%20the%20median%20first,be%20considered%20as%20the%20median.
void array_sort(double *array, int n)
{
    // declare some local variables
    int i = 0, j = 0;
    double temp = 0;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n - 1; j++)
        {
            if (array[j] > array[j + 1])
            {
                temp = array[j];
                array[j] = array[j + 1];
                array[j + 1] = temp;
            }
        }
    }
}

void print_array(float *vals, int num_elem)
{
    for (int i = 0; i < num_elem; i++)
    {
        printf("elem[%d]=%f\n", i, vals[i]);
    }
    printf("next\n");
}

double run_gemv_tblis(int dim)
{
    clock_t start, end;
    double cpu_time_used_ms;

    float *A_vals = malloc(sizeof(float) * dim * dim);
    float *b_vals = malloc(sizeof(float) * dim);
    float *c_vals = malloc(sizeof(float) * dim);

    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            A_vals[i * dim + j] = 2;
        }
    }
    for (int i = 0; i < dim; i++)
    {
        b_vals[i] = 1;
    }
    for (int i = 0; i < dim; i++)
    {
        c_vals[i] = 0;
    }

    tblis_tensor var1;
    tblis_tensor var2;
    tblis_tensor result;
    tblis_init_tensor_s(&var1, 2, (len_type[]){dim, dim}, A_vals, (stride_type[]){1, dim});
    tblis_init_tensor_s(&var2, 1, (len_type[]){dim}, b_vals, (stride_type[]){1});
    tblis_init_tensor_s(&result, 1, (len_type[]){dim}, c_vals, (stride_type[]){1});

    start = clock();
    tblis_tensor_mult(NULL, NULL, &var1, "ij", &var2, "j", &result, "i");
    end = clock();

    cpu_time_used_ms = ((double)(end - start)) / (CLOCKS_PER_SEC / 1000);
    return cpu_time_used_ms;
}

int main(int argc, char *argv[])
{
    for (int i = 100; i <= 5000; i += 100)
    {
        double time_taken[11];
        for (int j = 0; j < 11; j++)
        {
            time_taken[j] = run_gemv_tblis(i);
        }
        array_sort(time_taken, 11);
        printf("%d=%f\n", i, time_taken[5]);
    }
}