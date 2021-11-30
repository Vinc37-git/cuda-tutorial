#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define N 100000000
#define ERR_MAX 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    // Declare pointers for host (CPU)
    float *out, *a, *b;

    // Declare pointers for device (GPU)
    float *out_d, *a_d, *b_d;

    // allocate memory on host (CPU)
    out = (float*) malloc(sizeof(float) * N);
    a   = (float*) malloc(sizeof(float) * N);
    b   = (float*) malloc(sizeof(float) * N);

    // Initialize arrays on host (CPU)
    for (int i=0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // allocate memory on device (GPU)
    cudaMalloc(&out_d, sizeof(float) * N);
    cudaMalloc(&a_d, sizeof(float) * N);
    cudaMalloc(&b_d, sizeof(float) * N);

    // Transfer input data from host (CPU) to device (GPU)
    cudaMemcpy(a_d, a, sizeof(float) * N, cudaMemcpyHostToDevice); 
    cudaMemcpy(b_d, b, sizeof(float) * N, cudaMemcpyHostToDevice); 

    // perform addition on GPU
    vector_add<<<1,1>>>(out_d, a_d, b_d, N);

    // Free Memory of a_d und a_b
    cudaFree(a_d);
    cudaFree(b_d);

    // Transfer output data from device (GPU) to host (CPU)
    cudaMemcpy(out, out_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Free Memory of out_d
    cudaFree(out_d);

    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < ERR_MAX);
    }

    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");
    
    free(a);
    free(b);
    free(out);

    cudaDeviceSynchronize();
}