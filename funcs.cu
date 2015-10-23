#include <stdlib.h>
#include <stdio.h>

////////////////////////////////////////////////////////////////
////////////////// COPY EVERYTHING BELOW HERE //////////////////
////////////////////////////////////////////////////////////////

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

int main(int argc, char** argv) {
  
  return 0;
}