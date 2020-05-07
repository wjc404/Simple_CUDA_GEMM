#include <stdio.h>
#define CUDA_CALLER(call) do{\
  cudaError_t cuda_ret = (call);\
  if(cuda_ret != cudaSuccess){\
    printf("CUDA Error at line %d in file %s\n", __LINE__, __FILE__);\
    printf("  Error message: %s\n", cudaGetErrorString(cuda_ret));\
    exit(1);\
  }\
}while(0)
#define CUDA_KERNEL_CALLER(kernel_call) do{\
  if(cudaPeekAtLastError() != cudaSuccess){\
    printf("A CUDA error occurred above line %d in file %s\n", __LINE__, __FILE__); exit(1);\
  }\
  kernel_call;\
  CUDA_CALLER(cudaPeekAtLastError());\
}while(0)
