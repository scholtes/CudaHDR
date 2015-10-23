#include <stdlib.h>
#include <stdio.h>

// Number of elements to put in the test array
#define TEST_SIZE 32

////////////////////////////////////////////////////////////////
////////////////// COPY EVERYTHING BELOW HERE //////////////////
////////////////////////////////////////////////////////////////

// Number of threads per block (1-d blocks)
#define BLOCK_WIDTH 8

// Perform a partial reduction 
// Only reduces per block, so this kernel may need to be called
// multiple times to generate a complete reduction
__global__ void reduceKernel(float* array,
                       const size_t array_size,
                       float (*op)(float, float))
{
  __shared__ float temp[BLOCK_WIDTH];
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int index = BLOCK_WIDTH * bx + tx;

  if(index < array_size) {
    temp[tx] = array[index];
  }

  __syncthreads();

  // Reduce
  for(int offset = BLOCK_WIDTH >> 1; offset > 0; offset >>= 1) {
    if(tx < offset) {
      temp[tx] = (*op)(temp[tx], temp[tx + offset]);
    }
    __syncthreads();
  }

  if(index < array_size) {
    array[BLOCK_WIDTH * bx] = temp[0];
  } 

}

// Create a histogram with atomics
__global__ void histogramKernel() {

}

// This performs and *exclusive* scan
__global__ void scanKernel() {

}

__device__ float addOp(float a, float b) {
  return a + b;
}

__device__ float minOp(float a, float b) {
  return a < b ? a : b;
}

__device__ float maxOp(float a, float b) {
  return a > b ? a : b;
}

// Non-kernel (host) functions

void reduce(float* d_array,
            const size_t array_size,
            float* result,
            float (*op)(float, float))
{
  float *d_array_copy;
  size_t mem_size = sizeof(float) * array_size;
  cudaMalloc((void**) &d_array_copy, mem_size);
  cudaMemcpy(d_array_copy, d_array, mem_size, cudaMemcpyDeviceToDevice);

  size_t numBlocks = 1 + ((array_size - 1) / BLOCK_WIDTH);
  reduceKernel<<<numBlocks, BLOCK_WIDTH>>>(d_array_copy, array_size, op);

  cudaMemcpy(result, d_array_copy, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_array_copy);
}

////////////////////////////////////////////////////////////////
//////////////// EXCLUDE EVERYTHING BELOW HERE /////////////////
////////////////////////////////////////////////////////////////

void generateAndCopyTestValues(float** h_A, float** d_A, size_t size) {
  unsigned int mem_size = sizeof(float) * size;
  *h_A = (float*)malloc(mem_size);
  cudaMalloc((void**) d_A, mem_size);

  for(int i = 0; i < size; i++) {
    (*h_A)[i] = i+1;
  }

  cudaMemcpy(*d_A, *h_A, mem_size, cudaMemcpyHostToDevice);
}

void prettyprint(float *h_A, size_t size) {
  // Lots of magic numbers
  if(size <= 16) {
    for(int i = 0; i < size; i++) {
      printf("%0.1f ", h_A[i]);
    }
  } else {
    for(int i = 0; i < 8; i++) {
      printf("%0.1f ", h_A[i]);
    }
    printf("... ");
    for(int i = 0; i < 8; i++) {
      printf("%0.1f ", h_A[size +i -8]);
    }
  }
  printf("\n");
}

int main(int argc, char** argv) {
  float *h_array;
  float reduce_result_add;
  float reduce_result_max;
  float reduce_result_min;
  float *d_array;
  
  generateAndCopyTestValues(&h_array, &d_array, TEST_SIZE);

  printf("h_array = ");
  prettyprint(h_array, TEST_SIZE);

  // Perform reduce
  reduce(d_array, TEST_SIZE, &reduce_result_add, addOp);
  reduce(d_array, TEST_SIZE, &reduce_result_min, minOp);
  reduce(d_array, TEST_SIZE, &reduce_result_max, maxOp);
  printf("reduce_result_add = %0.1f\n", reduce_result_add);
  printf("reduce_result_min = %0.1f\n", reduce_result_min);
  printf("reduce_result_max = %0.1f\n", reduce_result_max);


  // Clean up
  free(h_array);
  cudaFree(d_array);
  return 0;
}