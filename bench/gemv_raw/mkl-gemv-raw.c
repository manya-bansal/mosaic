#include <stdio.h>
#include <time.h>
#include "mkl.h"

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

double run_gemv_mkl(int dim)
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
			A_vals[i * dim + j] = i + j;
		}
	}
	for (int i = 0; i < dim; i++)
	{
		b_vals[i] = i;
	}
	for (int i = 0; i < dim; i++)
	{
		c_vals[i] = 0;
	}
	float alpha = 1;
	MKL_INT beta = 1;
	MKL_INT inc = 1;
	MKL_INT two = dim;
	MKL_INT dimMkl = (MKL_INT)dim;
	start = clock();
	ssymv("u", &two, &alpha, A_vals, &dimMkl, b_vals, &inc, &alpha, c_vals, &inc);
	end = clock();
	cpu_time_used_ms = ((double)(end - start)) / (CLOCKS_PER_SEC / 1000);
	free(A_vals);
    free(c_vals);
    free(b_vals);
	return cpu_time_used_ms;
}

int main(int argc, char *argv[])
{	
	for (int i = 16; i <= 16; i += 100)
	{
		double time_taken[11];
		for (int j = 0; j < 11; j++)
		{
			time_taken[j] = run_gemv_mkl(i);
		}
		array_sort(time_taken, 11);
		printf("%d=%f\n", i, time_taken[5]);
	}
	// mkl_scsrgemv("n", (MKL_INT[]){0, 1}, (float[]){1}, (MKL_INT[]){0, 1}, (MKL_INT[]){0},  (float[]){0, 1},  (float[]){0, 0});
}