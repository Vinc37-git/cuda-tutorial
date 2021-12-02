#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1000
#define ROWS 15000
#define COLS 15000
#define MAX_ERR 1e-6

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

size_t ij(int i, int j){
    // Row Major
    return i * ROWS + j;
}

void vector_add(float *out, float *a, float *b, int n) {
    for (int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

void matrix_multi_elemwise(float *out, float *a, float *b, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[i,j] = a[i,j] * b[i,j];
        }
    }
}

int main(){
    float *a, *b, *out; 

    int n = ROWS * COLS, length = 0;

    // Allocate memory
    a   = (float*) malloc(sizeof(float) * n);
    b   = (float*) malloc(sizeof(float) * n);
    out = (float*) malloc(sizeof(float) * n);

    // Initialize array
    for(int i = 0; i < ROWS; i++){
        for (int j = 0; j < COLS; j++) {
            a[i,j] = 2.0f;
            b[i,j] = 3.0f;
            length++;
        }
    }
    printf("Length is %i elements.\n", length);

    // Main function
    matrix_multi_elemwise(out, a, b, ROWS, COLS);

    // Verification
    for(int i = 0; i < ROWS; i++){
        for (int j = 0; j < COLS; j++) {
            assert(fabs(out[i,j] / a[i,j] - b[i,j]) < MAX_ERR);
        }
    }

    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");
}
