#include <stdlib.h>
#include <stdio.h>

// Number of elements to put in the test array
#define TEST_SIZE 16
#define NUM_BINS 10

////////////////////////////////////////////////////////////////
////////////////// COPY EVERYTHING BELOW HERE //////////////////
////////////////////////////////////////////////////////////////

// Number of threads per block (1-d blocks)
#define BLOCK_WIDTH 4
// Functions to reduce with
#define ADD 0
#define MIN 1
#define MAX 2
// Device functions

__device__ float addOp(float a, float b) {
  return a + b;
}

__device__ float minOp(float a, float b) {
  return a < b ? a : b;
}

__device__ float maxOp(float a, float b) {
  return a > b ? a : b;
}

// Perform a partial reduction 
// Only reduces per block, so this kernel may need to be called
// multiple times to generate a complete reduction
__global__ void reduceKernel(float* array,
                       const size_t array_size,
                       const unsigned int op,
                       const size_t step)
{
  __shared__ float temp[BLOCK_WIDTH];
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int index = BLOCK_WIDTH * bx + tx;

  if(index < array_size) {
    temp[tx] = array[index * step];
  }

  __syncthreads();

  // Reduce
  for(int offset = BLOCK_WIDTH >> 1; offset > 0; offset >>= 1) {
    if(tx < offset) {
        switch(op) {
        case ADD:
          temp[tx] = addOp(temp[tx], temp[tx + offset]);
          break;
        case MIN:
          temp[tx] = minOp(temp[tx], temp[tx + offset]);
          break;
        case MAX:
          temp[tx] = maxOp(temp[tx], temp[tx + offset]);
          break;
        default:
          break;
        }
    }
    __syncthreads();
  }

  if(index < array_size) {
    array[BLOCK_WIDTH * bx] = temp[0];
  } 

}

// Create a histogram with atomics
__global__ void histogramKernel(unsigned int* d_hist,
                                const float* const d_array,
                                const size_t array_size,
                                float max,
                                float min,
                                const size_t numBins)
{
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int index = BLOCK_WIDTH * bx + tx;
  float range = max - min;

  // Initialize temp
  if(index < numBins) {
    d_hist[index] = 0;
  }

  __syncthreads();

  if(index < array_size) {
    size_t bin = (size_t)((d_array[index] - min) * numBins / range );
    atomicAdd(&d_hist[bin], 1);
  }
}

// This performs a partial exclusive scan (blockwise) using Blelloch's method
__global__ void scanKernel(unsigned int* d_cdf,
                           unsigned int* d_input,
                           const size_t array_size)
{
  __shared__ unsigned int temp[BLOCK_WIDTH<<1];
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int index = BLOCK_WIDTH * bx + tx;
  int offset = 1;

  if(2*index + 1 < array_size) {
    temp[2*index] = d_input[2*index];
    temp[2*index + 1] = d_input[2*index + 1];
  }
  
  // Up-sweep
  for(int powOf2 = (2*BLOCK_WIDTH)>>1; powOf2 > 0; powOf2 >>= 1) {
    __syncthreads();
    if(tx < powOf2) {
      int idx1 = offset*(2*tx + 1) - 1 + 2*BLOCK_WIDTH*bx;
      int idx2 = offset*(2*tx + 2) - 1 + 2*BLOCK_WIDTH*bx;
      temp[idx2] += temp[idx1];
    }
    offset <<= 1;
  }

  __syncthreads();
  ///// The below will need to be remembered for multiple blocks /////
  if(tx == 0) {
    temp[2*BLOCK_WIDTH*(bx + 1) - 1] = 0;
  }

  // Down-sweep
  for(int powOf2 = 1; powOf2 < 2*BLOCK_WIDTH; powOf2 <<= 1) {
    offset >>= 1;
    __syncthreads();
    if(tx < powOf2) {
      int idx1 = offset*(2*tx + 1) - 1 + 2*BLOCK_WIDTH*bx;
      int idx2 = offset*(2*tx + 2) - 1 + 2*BLOCK_WIDTH*bx;
      unsigned int t = temp[idx1];
      temp[idx1] = temp[idx2];
      temp[idx2] += t;
    }
  }

  __syncthreads();
  if(2*index + 1 < array_size) {
    d_cdf[2*index] = temp[2*index];
    d_cdf[2*index + 1] = temp[2*index + 1];
  }
}

void reduce(float* d_array,
            const size_t array_size,
            float* result,
            unsigned int op)
{
  float *d_array_copy;
  size_t mem_size = sizeof(float) * array_size;
  cudaMalloc((void**) &d_array_copy, mem_size);
  cudaMemcpy(d_array_copy, d_array, mem_size, cudaMemcpyDeviceToDevice);

  // First pass: 
  size_t numBlocks = 1 + ((array_size - 1) / BLOCK_WIDTH);
  reduceKernel<<<numBlocks, BLOCK_WIDTH>>>(d_array_copy, array_size, op, 1);
  // Second pass:
  reduceKernel<<<1, BLOCK_WIDTH>>>(d_array_copy, array_size, op, BLOCK_WIDTH);


  cudaMemcpy(result, d_array_copy, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_array_copy);
}

//d_hist, d_array, TEST_SIZE, reduce_result_max, reduce_result_min, numBins
void histogram(unsigned int** d_hist,
               const float* const d_array,
               const size_t array_size,
               float max,
               float min,
               const size_t numBins)
{
  cudaMalloc((void**) d_hist, sizeof(unsigned int) * numBins);

  size_t numBlocks = 1 + ((array_size - 1) / BLOCK_WIDTH);
  histogramKernel<<<numBlocks, BLOCK_WIDTH>>>(
      *d_hist,
      d_array,
      array_size,
      max,
      min,
      numBins
  );
}

void scan(unsigned int** d_cdf,
          unsigned int* d_input,
          const size_t array_size)
{
  cudaMalloc((void**) d_cdf, sizeof(unsigned int) * array_size);

  // Note the divide by 2 (a block can handle array of size 2*BLOCK_WIDTH)
  size_t numBlocks = (1 + ((array_size - 1) / BLOCK_WIDTH))/2;
  scanKernel<<<numBlocks, BLOCK_WIDTH>>>(*d_cdf, d_input, array_size);
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

void genTestValsCDF(unsigned int** h_A, unsigned int** d_A, size_t size) {
  unsigned int mem_size = sizeof(unsigned int) * size;
  *h_A = (unsigned int*)malloc(mem_size);
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

void prettyprint(unsigned int *h_A, size_t size) {
  // Lots of magic numbers
  if(size <= 16) {
    for(int i = 0; i < size; i++) {
      printf("%d ", h_A[i]);
    }
  } else {
    for(int i = 0; i < 8; i++) {
      printf("%d ", h_A[i]);
    }
    printf("... ");
    for(int i = 0; i < 8; i++) {
      printf("%d ", h_A[size +i -8]);
    }
  }
  printf("\n");
}

int main(int argc, char** argv) {
  // Reduce
  float *h_array;
  float *d_array;
  float reduce_result_add;
  float reduce_result_max;
  float reduce_result_min;
  // Histogram
  unsigned int *h_hist;
  unsigned int *d_hist;
  size_t numBins = NUM_BINS;
  // CDF (exclusive scan - prefix sum)
  unsigned int *h_toBeScanned;
  unsigned int *d_toBeScanned;
  unsigned int *h_cdf;
  unsigned int *d_cdf;
  
  generateAndCopyTestValues(&h_array, &d_array, TEST_SIZE);

  printf("h_array = ");
  prettyprint(h_array, TEST_SIZE);

  // Perform reduce
  reduce(d_array, TEST_SIZE, &reduce_result_add, ADD);
  reduce(d_array, TEST_SIZE, &reduce_result_min, MIN);
  reduce(d_array, TEST_SIZE, &reduce_result_max, MAX);
  printf("reduce_result_add = %0.1f\n", reduce_result_add);
  printf("reduce_result_min = %0.1f\n", reduce_result_min);
  printf("reduce_result_max = %0.1f\n", reduce_result_max);

  // Perform histogram
  histogram(&d_hist, d_array, TEST_SIZE, reduce_result_max, reduce_result_min, numBins);
  // Host histogram (not to be used in student_func)
  h_hist = (unsigned int*)malloc(sizeof(unsigned int) * numBins);
  cudaMemcpy(h_hist, d_hist, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost);
  printf("h_hist = ");
  prettyprint(h_hist, numBins);

  // Set up and perform exclusive scan
  genTestValsCDF(&h_toBeScanned, &d_toBeScanned, TEST_SIZE);

  printf("\nh_toBeScanned = ");
  prettyprint(h_toBeScanned, TEST_SIZE);

  scan(&d_cdf, d_toBeScanned, TEST_SIZE);
  h_cdf = (unsigned int*)malloc(sizeof(unsigned int) * TEST_SIZE);
  cudaMemcpy(h_cdf, d_cdf, sizeof(unsigned int) * TEST_SIZE, cudaMemcpyDeviceToHost);
  printf("h_cdf = ");
  prettyprint(h_cdf, TEST_SIZE);


  // Clean up
  free(h_array);
  free(h_hist);
  free(h_cdf);
  free(h_toBeScanned);
  cudaFree(d_array);
  cudaFree(d_hist);
  cudaFree(d_cdf);
  cudaFree(d_toBeScanned);
  return 0;
}