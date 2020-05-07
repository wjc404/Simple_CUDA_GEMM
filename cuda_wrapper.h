#include <stdio.h>
#define CUDA_CALLER(call) do{\
  cudaError_t cuda_ret = (call);\
  if(cuda_ret != cudaSuccess){\
    printf("CUDA Error at line %d in file %s\n", __LINE__, __FILE__);\
    printf("  Error message: %s\n", cudaGetErrorString(cuda_ret));\
    printf("  In the function call %s\n", #call);\
    exit(1);\
  }\
}while(0)
#define CUDA_KERNEL_CALLER(...) do{\
  if(cudaPeekAtLastError() != cudaSuccess){\
    printf("A CUDA error occurred prior to the kernel call %s at line %d\n", #__VA_ARGS__,  __LINE__); exit(1);\
  }\
  __VA_ARGS__;\
  cudaError_t cuda_ret = cudaPeekAtLastError();\
  if(cuda_ret != cudaSuccess){\
    printf("CUDA Error at line %d in file %s\n", __LINE__, __FILE__);\
    printf("  Error message: %s\n", cudaGetErrorString(cuda_ret));\
    printf("  In the kernel call %s\n", #__VA_ARGS__);\
    exit(1);\
  }\
}while(0)
