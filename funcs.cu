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
__global__ void reduce(float* array,
                       const size_t array_size,
                       float* result,
                       float (*op)(float, float))
{

}

// Create a histogram with atomics
__global__ void histogram() {

}

// This performs and *exclusive* scan
__global__ void scan() {

}

////////////////////////////////////////////////////////////////
//////////////// EXCLUDE EVERYTHING BELOW HERE /////////////////
////////////////////////////////////////////////////////////////

void generateAndCopyTestValues(float** h_A, float** d_A, float** d_A_copy, size_t size) {
  unsigned int mem_size = sizeof(float) * size;
  *h_A = (float*)malloc(mem_size);
  cudaMalloc((void**) d_A, mem_size);
  cudaMalloc((void**) d_A_copy, mem_size);

  for(int i = 0; i < size; i++) {
    (*h_A)[i] = i+1;
  }

  cudaMemcpy(*d_A, *h_A, mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(*d_A_copy, *h_A, mem_size, cudaMemcpyHostToDevice);
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
  float *d_array;
  float *d_array_copy; // Pass into reduce (which mutates the array)
  
  generateAndCopyTestValues(&h_array, &d_array, &d_array_copy, TEST_SIZE);

  printf("h_array = ");
  prettyprint(h_array, TEST_SIZE);

  return 0;
}