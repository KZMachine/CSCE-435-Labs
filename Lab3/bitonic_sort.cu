/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* Every thread gets exactly one value in the unsorted array. */
#define THREADS 64 // 2^9
#define BLOCKS 262144 // 2^15
#define NUM_VALS THREADS*BLOCKS

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %fs\n", elapsed);
}

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //MEM COPY FROM HOST TO DEVICE
  // clock_t startTime = clock();
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  // clock_t endTime = clock();
  // printf("Host to device: ");
  // print_elapsed(startTime, endTime);

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */
  
  int j, k;
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);


  clock_t kernalStartTime = clock();
  // cudaEventRecord(start);
  /* Major step */
  cudaEventRecord(start);

  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
    }
  }
  cudaEventRecord(stop);
  // cudaEventRecord(stop);
  cudaDeviceSynchronize();
  clock_t kernalEndTime = clock();
  printf("Kernel time: ");
  print_elapsed(kernalStartTime, kernalEndTime);
  
  //MEM COPY FROM DEVICE TO HOST
  // startTime = clock();
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  // endTime = clock();
  
  // printf("Device to host: ");
  // print_elapsed(startTime, endTime);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  float effectiveBW = NUM_VALS*sizeof(float)*2/milliseconds/1e6;
  printf("effective Bandwidth(Gb/s): %f", effectiveBW);
  cudaFree(dev_values);
}

int main(void)
{
  printf("Threads: %d, Blocks: %d\n", THREADS, BLOCKS);
  clock_t start, stop;

  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS);

  start = clock();
  bitonic_sort(values); /* Inplace */
  stop = clock();

  print_elapsed(start, stop);
}