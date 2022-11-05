#include <stdio.h>
#include <time.h>
#include "mkl.h"

// // from https://www.includehelp.com/c-programs/calculate-median-of-an-array.aspx#:~:text=To%20calculate%20the%20median%20first,be%20considered%20as%20the%20median.
// void array_sort(double *array , int n)
// { 
//     // declare some local variables
//     int i=0 , j=0 ;
//     double temp=0;

//     for(i=0 ; i<n ; i++)
//     {
//         for(j=0 ; j<n-1 ; j++)
//         {
//             if(array[j]>array[j+1])
//             {
//                 temp        = array[j];
//                 array[j]    = array[j+1];
//                 array[j+1]  = temp;
//             }
//         }
//     }

// }

// double run_gemv_tblis(int dim){
//     clock_t start, end;
//     double cpu_time_used_ms;

//     float * A_vals = malloc(sizeof(float)*dim*dim);
//     float * b_vals = malloc(sizeof(float)*dim);
//     float * c_vals = malloc(sizeof(float)*dim);

//     for(int i = 0; i < dim; i++){
//         for(int j = 0; j < dim; j++){
//             A_vals[i*dim +j] = i+j;
//         }
//     }
//     for(int i = 0; i < dim; i++){
//         b_vals[i] = i;
//     }
//     for(int i = 0; i < dim; i++){
//         c_vals[i] = 0;
//     }

//     tblis_tensor var1;
//     tblis_tensor var2;
//     tblis_tensor result;
//     tblis_init_tensor_s(&var1, 2, (len_type[]){dim, dim}, A_vals, (stride_type[]){1, dim});
//     tblis_init_tensor_s(&var2, 1, (len_type[]){dim}, b_vals, (stride_type[]){1});
//     tblis_init_tensor_s(&result, 1, (len_type[]){dim}, c_vals, (stride_type[]){1});
    
//     start = clock();
//     tblis_tensor_mult(NULL, NULL, &var1, "ij", &var2, "j", &result, "i");
//     end = clock();

//     cpu_time_used_ms = ((double) (end - start)) / (CLOCKS_PER_SEC/1000);
//     return cpu_time_used_ms;
// }

int main(int argc, char *argv[]) {
//    for (int i = 100; i<=1000; i+=100){
//         // double time_taken[11];
//         // for (int j = 0; j<11; j++){
//         //     time_taken[j] = run_gemv_tblis(i);
//         // }
//         // array_sort(time_taken, 11);
//         printf("%d=%f\n", i, time_taken[5]);
//     }
        mkl_scsrgemv("n", (MKL_INT[]){0, 1}, (float[]){1}, (MKL_INT[]){0, 1}, (MKL_INT[]){0},  (float[]){0, 1},  (float[]){0, 0});
        printf("compiles!\n =)");
    
}