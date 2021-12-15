#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1000
#define ROWS 256//16384
#define COLS 256
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

// __global__ void vector_add(float *out, float *a, float *b, int n) {
//     // threadIdx.x contains the index of the thread within the block
//     // blockDim.x contains the size of thread block (number of threads in the thread block).

//     // blockIdx.x contains the index of the block with in the grid
//     // gridDim.x contains the size of the grid

//     int index = (blockIdx.x * blockDim.x) + threadIdx.x;  // linearisation of index tuple
//     int stride = gridDim.x * blockDim.x;  // 
//     for(int i = index /*"range" for every open thread*/; i < n; i += stride /* e.g + 256*/){
//         out[i] = a[i] + b[i];
//     }
// }

// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
//     #define printf(f, ...) ((void)(f, __VA_ARGS__),0) 
// #endif 

__global__ void matrix_multi_elemwise(Matrix OUT, const Matrix A, const Matrix B) {
    // NOTE: Generally memory allocated dynamically on device (GPU) and 
    // we cannot use two-dimensional indices (e.g. A[row][column]) 
    // to access matrices -> linear indexing
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    //printf("%d, %d\n", col, row);
    //printf("%f\n", A.elements[1]);

    int index = row * A.width + col;  // linearisation of index

    if (col < A.width && row < A.height) {
        OUT.elements[index] = A.elements[index] * B.elements[index];
    }
}

int main(){
    printf("Start\n");
    Matrix A, B, OUT;
    Matrix dev_A, dev_B, dev_OUT; 

    size_t SIZE = ROWS * COLS * sizeof(float);

    // Allocate host memory
    A.elements = (float*) malloc(SIZE);
    B.elements = (float*) malloc(SIZE);
    OUT.elements = (float*) malloc(SIZE);

    // Initialize host matrices
    A.height = ROWS; A.width = COLS;
    B.height = ROWS; B.width = COLS;
    OUT.height = ROWS; OUT.width = COLS;

    for (int i = 0; i < ROWS; i++) {
        for(int j = 0; j < COLS; j++){
            A.elements[ij(i, j)] = 2.0f;
            B.elements[ij(i, j)] = 3.0f;
        }
    }

    // Allocate device memory
    cudaMalloc((void**) &dev_A.elements, SIZE);
    cudaMalloc((void**) &dev_B.elements, SIZE);
    cudaMalloc((void**) &dev_OUT.elements, SIZE);

    dev_A.height = A.height; dev_A.width = A.width;
    dev_B.height = A.height; dev_B.width = B.width;
    dev_OUT.height = A.height; dev_OUT.width = OUT.width;

    // Transfer data from host to device memory
    cudaMemcpy(dev_A.elements, A.elements, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B.elements, B.elements, SIZE, cudaMemcpyHostToDevice);

    // Executing kernel 
    const dim3 threads(32, 32);
    const dim3 blocks((COLS + threads.x - 1) / threads.x, (ROWS + threads.y - 1) / threads.y);
    if (threads.x * blocks.x < ROWS || threads.y * blocks.y < COLS) {
        printf("Program terminated. Block dim: %i, %i, Grid dim: %i, %i, Total threads: %i, %i.\n", threads.x, threads.y, blocks.x, blocks.y, threads.x * blocks.x, threads.y * blocks.y);
        return 0;
    }
    matrix_multi_elemwise<<<blocks, threads>>>(dev_OUT, dev_A, dev_B);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("CUDA Runtime API Error reported : %s in file %s on line.\n", cudaGetErrorString(err), __FILE__);
    }
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();  
    
    // Transfer data back to host memory
    cudaMemcpy(OUT.elements, dev_OUT.elements, SIZE, cudaMemcpyDeviceToHost);

    // Verification
    int count = 0, length = 0, i = 0, j = 0;
    for (i = 0; i < ROWS; i++) {
        for(j = 0; j < COLS; j++){
            //assert(fabs(OUT.elements[ij(i, j)] / A.elements[ij(i, j)] - B.elements[ij(i, j)]) < MAX_ERR);
            if (fabs(OUT.elements[ij(i, j)] / A.elements[ij(i, j)] - B.elements[ij(i, j)]) > MAX_ERR) {
                count++;
            }
            length++;
        }
    }
    printf("Verification: %i elements have failed, total length %i, shape: (%i, %i).\n", count, length, i, j);

    // Deallocate device memory
    cudaFree(dev_A.elements);
    cudaFree(dev_B.elements);
    cudaFree(dev_OUT.elements);
    
    // Deallocate host memory
    free(A.elements); 
    free(B.elements); 
    free(OUT.elements);
    
    // flush profile data
    // cuProfilerStop();
}
