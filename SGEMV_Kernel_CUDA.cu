//nvcc -arch=sm_35 -Xcompiler -fopenmp SGEMV_Kernel_CUDA.cu -o SGEMV_TEST
#include "cuda_wrapper.h"
#define BlkDim_Y 8
#define GEMV1_BlkDim_K 1024
__global__ void gemv1_kernel_general(const float * __restrict__ A_vec0,
  const float * __restrict__ B_mat0, float * __restrict__ C_vec0,
  unsigned int N, unsigned int K) { // matrix B row-major
  const float * const A_vec = A_vec0 + K * blockIdx.x;
  const float * const B_mat = B_mat0 + K * N * blockIdx.x;
  float * const C_vec = C_vec0 + N * blockIdx.x;
  __shared__ float a_cache[GEMV1_BlkDim_K];
  unsigned int n_pos, k_pos1, k_inc1, k_pos2;
  float a0, b0, c0; const float *B_ptr; float *a_ptr;
  for (k_pos1 = 0; k_pos1 < K; k_pos1 += k_inc1) {
    k_inc1 = K - k_pos1; if (k_inc1 > GEMV1_BlkDim_K) k_inc1 = GEMV1_BlkDim_K;
    for (k_pos2 = threadIdx.x + threadIdx.y * blockDim.x; k_pos2 < k_inc1;
      k_pos2 += blockDim.y * blockDim.x) {
      a_cache[k_pos2] = A_vec[k_pos2 + k_pos1];
    }
    __syncthreads();
    for (n_pos = threadIdx.x + threadIdx.y * blockDim.x; n_pos < N;
      n_pos += blockDim.y * blockDim.x) {
      B_ptr = B_mat + n_pos + k_pos1 * N; a_ptr = a_cache;
      c0 = 0.0f;
      for (k_pos2 = k_inc1; k_pos2 > 3; k_pos2 -= 4) {
        a0 = *a_ptr; a_ptr++; b0 = *B_ptr; B_ptr += N;
        c0 += a0 * b0;
        a0 = *a_ptr; a_ptr++; b0 = *B_ptr; B_ptr += N;
        c0 += a0 * b0;
        a0 = *a_ptr; a_ptr++; b0 = *B_ptr; B_ptr += N;
        c0 += a0 * b0;
        a0 = *a_ptr; a_ptr++; b0 = *B_ptr; B_ptr += N;
        c0 += a0 * b0;
      }
      for (; k_pos2 > 0; k_pos2--) {
        a0 = *a_ptr; a_ptr++; b0 = *B_ptr; B_ptr += N;
        c0 += a0 * b0;
      }
      C_vec[n_pos] += c0;
    }
    __syncthreads();
  }
}
#define GEMV2_BlkDim_K 1024
__global__ void gemv2_kernel_general(const float * __restrict__ A_vec0,
  const float * __restrict__ B_mat0, float * __restrict__ C_vec0,
  unsigned int N, unsigned int K) { // matrix B column-major
  //blockDim.x must be 32 for this function!
  const float * const A_vec = A_vec0 + K * blockIdx.x;
  const float * const B_mat = B_mat0 + K * N * blockIdx.x;
  float * const C_vec = C_vec0 + N * blockIdx.x;
  __shared__ float a_cache[GEMV2_BlkDim_K];
  unsigned int n_pos, k_pos1, k_inc1, k_pos2, k_upper;
  const unsigned int n_start = (N * threadIdx.y) / blockDim.y;
  const unsigned int n_end = (N * (threadIdx.y + 1)) / blockDim.y;
  float a0, b0, c0; const float *B_ptr; float *a_ptr;
  for (k_pos1 = 0; k_pos1 < K; k_pos1 += k_inc1) {
    k_inc1 = K - k_pos1; if (k_inc1 > GEMV2_BlkDim_K) k_inc1 = GEMV2_BlkDim_K;
    for (k_pos2 = threadIdx.x + threadIdx.y * blockDim.x; k_pos2 < k_inc1;
      k_pos2 += blockDim.y * blockDim.x) {
      a_cache[k_pos2] = A_vec[k_pos2 + k_pos1];
    }
    __syncthreads();
    k_upper = (k_inc1 > 3 * blockDim.x) ? (k_inc1 - 3 * blockDim.x) : 0;
    for (n_pos = n_start; n_pos < n_end; ++n_pos) {
      B_ptr = B_mat + n_pos * K + k_pos1 + threadIdx.x; c0 = 0.0f;
      a_ptr = a_cache + threadIdx.x;
      for (k_pos2 = threadIdx.x; k_pos2 < k_upper; k_pos2 += blockDim.x * 4) {
        a0 = *a_ptr; a_ptr += blockDim.x;
        b0 = *B_ptr; B_ptr += blockDim.x;
        c0 += a0 * b0;
        a0 = *a_ptr; a_ptr += blockDim.x;
        b0 = *B_ptr; B_ptr += blockDim.x;
        c0 += a0 * b0;
        a0 = *a_ptr; a_ptr += blockDim.x;
        b0 = *B_ptr; B_ptr += blockDim.x;
        c0 += a0 * b0;
        a0 = *a_ptr; a_ptr += blockDim.x;
        b0 = *B_ptr; B_ptr += blockDim.x;
        c0 += a0 * b0;
      }
      for (; k_pos2 < k_inc1; k_pos2 += blockDim.x) {
        a0 = *a_ptr; a_ptr += blockDim.x;
        b0 = *B_ptr; B_ptr += blockDim.x;
        c0 += a0 * b0;
      }
      c0 += __shfl_down(c0, 16);
      c0 += __shfl_down(c0, 8);
      c0 += __shfl_down(c0, 4);
      c0 += __shfl_down(c0, 2);
      c0 += __shfl_down(c0, 1);
      if(!threadIdx.x) C_vec[n_pos] += c0;
    }
    __syncthreads();
  }
}
__host__ void batch_gemv_gpu(const float * __restrict__ d_A_vec,
  const float * __restrict__ d_B_mat, float * __restrict__ d_C_vec,
  unsigned int batch, unsigned int N, unsigned int K,
  unsigned int browmajorflag) {
  dim3 block_dim(32,BlkDim_Y);
  if (browmajorflag) {
    CUDA_KERNEL_CALLER(
      gemv1_kernel_general<<<batch, block_dim>>>
        (d_A_vec, d_B_mat, d_C_vec, N, K));
  } else {
    CUDA_KERNEL_CALLER(
      gemv2_kernel_general<<<batch, block_dim>>>
        (d_A_vec, d_B_mat, d_C_vec, N, K));
  }
  CUDA_CALLER(cudaDeviceSynchronize());
}
__host__ void gemv1_kernel_general_h(const float * __restrict__ A_vec,
  const float * __restrict__ B_mat, float * __restrict__ C_vec,
  unsigned int N, unsigned int K) { // matrix B row-major
  unsigned int n_pos, k_pos;
  const float *B_ptr1 = B_mat, *B_ptr2 = B_mat + N,
    *B_ptr3 = B_mat + N * 2, *B_ptr4 = B_mat + N * 3, *A_ptr = A_vec;
  unsigned int B_inc = 4 * N; float a0, a1, a2, a3, c0;
  for (k_pos = K; k_pos > 3; k_pos -= 4) {
    a0 = A_ptr[0]; a1 = A_ptr[1];
    a2 = A_ptr[2]; a3 = A_ptr[3]; A_ptr += 4;
    for (n_pos = 0; n_pos < N; ++n_pos) {
      c0 = B_ptr1[n_pos] * a0;
      c0 += B_ptr2[n_pos] * a1;
      c0 += B_ptr3[n_pos] * a2;
      c0 += B_ptr4[n_pos] * a3;
      C_vec[n_pos] += c0;
    }
    B_ptr1 += B_inc; B_ptr2 += B_inc; B_ptr3 += B_inc; B_ptr4 += B_inc;
  }
  for (; k_pos > 0; k_pos--) {
    a0 = *A_ptr; A_ptr++;
    for (n_pos = 0; n_pos < N; ++n_pos) {
      C_vec[n_pos] += B_ptr1[n_pos] * a0;
    }
    B_ptr1 += N;
  }
}
__host__ void gemv2_kernel_general_h(const float * __restrict__ A_vec,
  const float * __restrict__ B_mat, float * __restrict__ C_vec,
  unsigned int N, unsigned int K) { // matrix B column-major
  unsigned int n_pos, k_pos, k_upper;
  k_upper = (K > 3) ? (K - 3) : 0;
  float c0; const float *B_ptr;
  for (n_pos = 0; n_pos < N; ++n_pos) {
    c0 = 0.0f; B_ptr = B_mat + n_pos * K;
    for (k_pos = 0; k_pos < k_upper; k_pos += 4) {
      c0 += A_vec[k_pos] * B_ptr[k_pos];
      c0 += A_vec[k_pos + 1] * B_ptr[k_pos + 1];
      c0 += A_vec[k_pos + 2] * B_ptr[k_pos + 2];
      c0 += A_vec[k_pos + 3] * B_ptr[k_pos + 3];
    }
    for (; k_pos < K; ++k_pos) {
      c0 += A_vec[k_pos] * B_ptr[k_pos];
    }
    C_vec[n_pos] += c0;
  }
}
__host__ void batch_gemv_cpu(const float * __restrict__ A_vec,
  const float * __restrict__ B_mat, float * __restrict__ C_vec,
  unsigned int batch, unsigned int N, unsigned int K,
  unsigned int browmajorflag) {
#pragma omp parallel for
  for(unsigned int batno = 0; batno < batch; ++batno) {
    if (browmajorflag) {
      gemv1_kernel_general_h(A_vec + batno * K, B_mat + batno * K * N,
        C_vec + batno * N, N, K);
    } else {
      gemv2_kernel_general_h(A_vec + batno * K, B_mat + batno * K * N,
        C_vec + batno * N, N, K);
    }
  }
}
#include <string.h>
#include <time.h>
#include <sys/time.h>
int main(int argc, char **argv) {
  if (argc > 1) {
    if (!strcmp(argv[1],"--help") || !strcmp(argv[1],"-H")) {
      printf("\t%s <batch> <N> <K>\n",argv[0]);
      return 0;
    }
  }
  unsigned int batch = 10, N = 1000, K = 1000;
  if (argc > 1) batch = atoi(argv[1]);
  if (argc > 2) N = atoi(argv[2]);
  if (argc > 3) K = atoi(argv[3]);
  printf("Information: batch = %u, N = %u, K = %u\n", batch, N, K);
  float * const h_A_vec = (float *)malloc(batch * K * sizeof(float));
  float * const h_B_mat = (float *)malloc(batch * K * N * sizeof(float));
  float * const h_C_vec1 = (float *)malloc(batch * N * sizeof(float));
  float * const h_C_vec2 = (float *)malloc(batch * N * sizeof(float));
  if (h_A_vec == NULL || h_B_mat == NULL ||
    h_C_vec1 == NULL || h_C_vec2 == NULL) {
    printf("Allocation of arrays on host memory failed. ");
    printf("Please try with smaller problem size.\n");
    return 1;
  }
  unsigned int count; srand(time(NULL));
  for (count = 0; count < batch * K; ++count) {
    h_A_vec[count] = (float)rand() / RAND_MAX;
  }
#pragma omp parallel for
  for (count = 0; count < batch * K * N; ++count) {
    h_B_mat[count] = (float)rand() / RAND_MAX;
  }
  for (count = 0; count < batch * N; ++count) {
    h_C_vec1[count] = h_C_vec2[count] = (float)rand() / RAND_MAX;
  }
  printf("Initialization of arrays on host memory: Done.\n");

  CUDA_CALLER(cudaSetDevice(0)); float *d_A_vec, *d_B_mat, *d_C_vec;
  CUDA_CALLER(cudaMalloc((void **)&d_A_vec, batch * K * sizeof(float)));
  CUDA_CALLER(cudaMalloc((void **)&d_B_mat, batch * K * N * sizeof(float)));
  CUDA_CALLER(cudaMalloc((void **)&d_C_vec, batch * N * sizeof(float)));
  CUDA_CALLER(cudaMemcpy(d_A_vec, h_A_vec,
    batch * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALLER(cudaMemcpy(d_B_mat, h_B_mat,
    batch * K * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALLER(cudaMemcpy(d_C_vec, h_C_vec1,
    batch * N * sizeof(float), cudaMemcpyHostToDevice));
  printf("Initialization of arrays on device memory: Done.\n");

  struct timeval start_time, end_time; double nsec; float tmp, max;
  printf("First test row-major matrices of B:\n");
  gettimeofday(&start_time, 0);
  batch_gemv_cpu(h_A_vec, h_B_mat, h_C_vec1, batch, N, K, 1);
  gettimeofday(&end_time, 0);
  nsec = 1.0e9 * (double)(end_time.tv_sec - start_time.tv_sec)
    + 1.0e3 * (double)(end_time.tv_usec - start_time.tv_usec);
  printf("\tCalculations on CPU: Done.\n");
  printf("\t\tBandwidth of reading matrix B: %.2e GB/s\n",
    (double)(sizeof(float) * K) * (double)(N * batch) / nsec);
  gettimeofday(&start_time, 0);
  batch_gemv_gpu(d_A_vec, d_B_mat, d_C_vec, batch, N, K, 1);
  gettimeofday(&end_time, 0);
  nsec = 1.0e9 * (double)(end_time.tv_sec - start_time.tv_sec)
    + 1.0e3 * (double)(end_time.tv_usec - start_time.tv_usec);
  printf("\tCalculations on GPU: Done.\n");
  printf("\t\tBandwidth of reading matrix B: %.2e GB/s\n",
    (double)(sizeof(float) * K) * (double)(N * batch) / nsec);
  CUDA_CALLER(cudaMemcpy(h_C_vec2, d_C_vec,
    batch * N * sizeof(float), cudaMemcpyDeviceToHost));
  max = 0.0f;
  for (count = 0; count < batch * N; ++count) {
    tmp = h_C_vec2[count] - h_C_vec1[count];
    if (tmp < 0) tmp *= -1.0;
    if (tmp > max) max = tmp;
  }
  printf("\tMax diff. between the results from host and device:");
  printf(" %.2e\n", max);
  memcpy(h_C_vec1, h_C_vec2, batch * N * sizeof(float));

  printf("Then test column-major matrices of B:\n");
  gettimeofday(&start_time, 0);
  batch_gemv_cpu(h_A_vec, h_B_mat, h_C_vec1, batch, N, K, 0);
  gettimeofday(&end_time, 0);
  nsec = 1.0e9 * (double)(end_time.tv_sec - start_time.tv_sec)
    + 1.0e3 * (double)(end_time.tv_usec - start_time.tv_usec);
  printf("\tCalculations on CPU: Done.\n");
  printf("\t\tBandwidth of reading matrix B: %.2e GB/s\n",
    (double)(sizeof(float) * K) * (double)(N * batch) / nsec);
  gettimeofday(&start_time, 0);
  batch_gemv_gpu(d_A_vec, d_B_mat, d_C_vec, batch, N, K, 0);
  gettimeofday(&end_time, 0);
  nsec = 1.0e9 * (double)(end_time.tv_sec - start_time.tv_sec)
    + 1.0e3 * (double)(end_time.tv_usec - start_time.tv_usec);
  printf("\tCalculations on GPU: Done.\n");
  printf("\t\tBandwidth of reading matrix B: %.2e GB/s\n",
    (double)(sizeof(float) * K) * (double)(N * batch) / nsec);
  CUDA_CALLER(cudaMemcpy(h_C_vec2, d_C_vec,
    batch * N * sizeof(float), cudaMemcpyDeviceToHost));
  max = 0.0f;
  for (count = 0; count < batch * N; ++count) {
    tmp = h_C_vec2[count] - h_C_vec1[count];
    if (tmp < 0) tmp *= -1.0;
    if (tmp > max) max = tmp;
  }
  printf("\tMax diff. between the results from host and device:");
  printf(" %.2e\n", max);

  CUDA_CALLER(cudaFree(d_A_vec)); CUDA_CALLER(cudaFree(d_B_mat));
  CUDA_CALLER(cudaFree(d_C_vec));
  free(h_A_vec); free(h_B_mat); free(h_C_vec1); free(h_C_vec2);
  return 0;
}

