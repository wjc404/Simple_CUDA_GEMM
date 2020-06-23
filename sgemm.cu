//nvcc --shared -Xcompiler -fPIC -Xptxas -O3,--verbose -arch=sm_60 sgemm.cu -o sgemm.so
#ifndef blockDim_X_half
 #define blockDim_X_half 16
#endif
#ifndef blockDim_Y_half
 #define blockDim_Y_half 2
#endif
#define blockDim_X (blockDim_X_half*2)
#define blockDim_Y (blockDim_Y_half*2)
#define blksize (blockDim_X*blockDim_Y)
#define share_ld_dim (blksize+2)
void __global__ sgemm_kernel(const float alpha, const float beta, const int LDA, const int LDB, const int LDC, const int AB_order,
 const int M, const int N, const int K, const float * __restrict__ A, const float * __restrict__ B, float *C){
  /* row_major matrix C; blockdim(blockDim_X, blockDim_Y) */
  const int Rowmajor_A = AB_order & 1;
  const int Rowmajor_B = AB_order & 2;
  register float2 c[blockDim_X][blockDim_Y_half];
  __shared__ __align__(8) float shared_a[share_ld_dim*blockDim_X], shared_b[share_ld_dim*blockDim_X];
  const int m_base = blockIdx.y * blksize;
  const int n_base = blockIdx.x * blksize;
#pragma unroll
  for(int i=0; i<blockDim_X; i++){
#pragma unroll
    for(int j=0; j<blockDim_Y_half; j++) c[i][j].x = c[i][j].y = 0.0;
  }
  float2 *sa = (float2 *)shared_a + threadIdx.y*blockDim_X_half, *sb = (float2 *)shared_b + threadIdx.x;
  register float2 a2[blockDim_X], b2[blockDim_Y_half];
  bool load_cond;
  const int k_major_mn_biase = threadIdx.y*blockDim_X;
  const int mn_major_mn_biase = k_major_mn_biase + threadIdx.x;
  float * const share_a_write = Rowmajor_A ? (shared_a+threadIdx.x*share_ld_dim+k_major_mn_biase) : (shared_a+mn_major_mn_biase);
  float * const share_b_write = Rowmajor_B ? (shared_b+threadIdx.y/2*(blockDim_X*2)+threadIdx.y%2+(threadIdx.x<<1)) : (shared_b+threadIdx.x*share_ld_dim+threadIdx.y%2+threadIdx.y/2*(blockDim_X*2));
  const int a_ptr_inc_last = Rowmajor_A ? (blockDim_X-blockDim_X*LDA) : 0;
  const int b_ptr_inc_last = Rowmajor_B ? 0 : (blockDim_X-blockDim_X*LDB);
  const float * __restrict__ a_ptr = Rowmajor_A ? (A+(m_base+k_major_mn_biase)*LDA+threadIdx.x) : (A+m_base+mn_major_mn_biase);
  const float * __restrict__ b_ptr = Rowmajor_B ? (B+n_base+mn_major_mn_biase) : (B+(n_base+k_major_mn_biase)*LDB+threadIdx.x);
  int load_cond_a_lhs = Rowmajor_A ? (threadIdx.x-K) : (m_base+mn_major_mn_biase-M);
  int load_cond_b_lhs = Rowmajor_B ? (n_base+mn_major_mn_biase-N) : (threadIdx.x-K);
  const int update_load_cond_a_lhs = Rowmajor_A ? blockDim_X : 0;
  const int update_load_cond_b_lhs = Rowmajor_B ? 0 : blockDim_X;
  int l2_cond_a_lhs = Rowmajor_A ? (m_base+k_major_mn_biase-M) : (-K);
  int l2_cond_b_lhs = Rowmajor_B ? (-K) : (n_base+k_major_mn_biase-N);
  const int update_l2_cond_a_lhs = Rowmajor_A ? 0 : blockDim_X;
  const int update_l2_cond_b_lhs = Rowmajor_B ? blockDim_X : 0;
  const int shared_a_wrptr_update = Rowmajor_A ? 1 : share_ld_dim;
  const int shared_b_wrptr_update = Rowmajor_B ? share_ld_dim : 2;
  for(int k_pos=0; k_pos<K; k_pos+=blockDim_X){
    load_cond = (load_cond_a_lhs<0);
    if(load_cond && l2_cond_a_lhs<=-blockDim_X){
#pragma unroll
      for(int ll=0; ll<blockDim_X; ll++){
        a2[ll].x = *a_ptr; a_ptr += LDA;
      }
    }else{
#pragma unroll
      for(int ll=0; ll<blockDim_X; ll++){
        if(l2_cond_a_lhs+ll<0 && load_cond) a2[ll].x = *a_ptr;
        else a2[ll].x = 0.0;
        a_ptr += LDA;
      }
    }
    a_ptr += a_ptr_inc_last;
    load_cond_a_lhs += update_load_cond_a_lhs; l2_cond_a_lhs += update_l2_cond_a_lhs;
    load_cond = (load_cond_b_lhs<0);
    if(load_cond && l2_cond_b_lhs<=-blockDim_X){
#pragma unroll
      for(int ll=0; ll<blockDim_X; ll++){
        a2[ll].y = *b_ptr; b_ptr += LDB;
      }
    }else{
#pragma unroll
      for(int ll=0; ll<blockDim_X; ll++){
        if(l2_cond_b_lhs+ll<0 && load_cond) a2[ll].y = *b_ptr;
        else a2[ll].y = 0.0;
        b_ptr += LDB;
      }
    }
    b_ptr += b_ptr_inc_last;
    load_cond_b_lhs += update_load_cond_b_lhs; l2_cond_b_lhs += update_l2_cond_b_lhs;
#pragma unroll
    for(int ll=0; ll<blockDim_X; ll++) share_a_write[ll*shared_a_wrptr_update] = a2[ll].x;
#pragma unroll
    for(int ll=0; ll<blockDim_X; ll++) share_b_write[ll*shared_b_wrptr_update] = a2[ll].y;
    __syncthreads();
#pragma unroll 2
    for(int ll=0; ll<blockDim_X; ll++){
#pragma unroll
      for(int i=0; i<blockDim_Y_half; i++)  b2[i] = sb[ll*(share_ld_dim/2)+i*blockDim_X];
#pragma unroll
      for(int i=0; i<blockDim_X_half; i++){
        a2[i] = sa[ll*(share_ld_dim/2)+i];
#pragma unroll
        for(int j=0; j<blockDim_Y_half; j++){
          c[i*2][j].x += a2[i].x * b2[j].x; c[i*2][j].y += a2[i].x * b2[j].y;
        }
#pragma unroll
        for(int j=0; j<blockDim_Y_half; j++){
          c[i*2+1][j].x += a2[i].y * b2[j].x; c[i*2+1][j].y += a2[i].y * b2[j].y;
        }
      }
    }
  __syncthreads();
  }
#pragma unroll
  for(int i=0; i<blockDim_X; i++){
#pragma unroll
    for(int j=0; j<blockDim_Y_half; j++){ c[i][j].x *= alpha; c[i][j].y *= alpha;}
  }
  int m_pos = m_base+threadIdx.y*blockDim_X, n_pos = 0;
  float *c_ptr = C + m_pos * LDC;
#pragma unroll
  for(int i=0; i<blockDim_X; i++){
    if(m_pos>=M) return;
    n_pos = n_base+threadIdx.x;
#pragma unroll
    for(int j=0; j<blockDim_Y_half; j++){
      if(n_pos<N){
        c[i][j].x += c_ptr[n_pos] * beta; c_ptr[n_pos] = c[i][j].x;
      }
      n_pos += blockDim_X;
      if(n_pos<N){
        c[i][j].y += c_ptr[n_pos] * beta; c_ptr[n_pos] = c[i][j].y;
      }
      n_pos += blockDim_X;
    }
    c_ptr += LDC; m_pos ++;
  }
}
#include "cuda_wrapper.h"
#include <sys/time.h>
#ifndef DEVICE_NO
 #define DEVICE_NO 0
#endif
extern "C" void sgemm_(const char *transa, const char *transb, const int *M, const int *N, const int *K,
 const float *alpha, const float *A, const int *LDA, const float *B, const int *LDB, const float *beta, float *C, const int *LDC){
  CUDA_CALLER(cudaSetDevice(DEVICE_NO)); size_t d_lda, d_ldb, d_ldc; int d_LDA, d_LDB, d_LDC;
  float *d_A, *d_B, *d_C; struct timeval starttime, endtime;
  int A_width, A_height, B_width, B_height;
  if(*transa == 'N'){ A_width = *M; A_height = *K;} else{ A_width = *K; A_height = *M;}
  if(*transb == 'N'){ B_width = *K; B_height = *N;} else{ B_width = *N; B_height = *K;}
#ifdef ALIGNED_ALLOC
  CUDA_CALLER(cudaMallocPitch((void **)&d_A, &d_lda, A_width * sizeof(float), A_height));
  CUDA_CALLER(cudaMallocPitch((void **)&d_B, &d_ldb, B_width * sizeof(float), B_height));
  CUDA_CALLER(cudaMallocPitch((void **)&d_C, &d_ldc, (*M) * sizeof(float), *N));
  d_LDA = d_lda/sizeof(float); d_LDB = d_ldb/sizeof(float); d_LDC = d_ldc/sizeof(float);
#else
  d_LDA = A_width; d_LDB = B_width; d_LDC = *M;
  d_lda = d_LDA * sizeof(float); d_ldb = d_LDB * sizeof(float); d_ldc = d_LDC * sizeof(float);
  CUDA_CALLER(cudaMalloc((void **)&d_A, d_lda  * A_height));
  CUDA_CALLER(cudaMalloc((void **)&d_B, d_ldb  * B_height));
  CUDA_CALLER(cudaMalloc((void **)&d_C, d_ldc  * (*N)));
#endif
  CUDA_CALLER(cudaMemcpy2D(d_A, d_lda, A, *LDA*sizeof(float), A_width*sizeof(float), A_height, cudaMemcpyHostToDevice));
  CUDA_CALLER(cudaMemcpy2D(d_B, d_ldb, B, *LDB*sizeof(float), B_width*sizeof(float), B_height, cudaMemcpyHostToDevice));
  CUDA_CALLER(cudaMemcpy2D(d_C, d_ldc, C, *LDC*sizeof(float), *M*sizeof(float), *N, cudaMemcpyHostToDevice));
  dim3 gemm_grid((*M-1)/blksize+1, (*N-1)/blksize+1), gemm_block(blockDim_X, blockDim_Y);
  int AB_order = 0; if(*transa == 'N') AB_order |= 2; if(*transb == 'N') AB_order |= 1;
  gettimeofday(&starttime,0);
  CUDA_KERNEL_CALLER(sgemm_kernel<<<gemm_grid, gemm_block>>>(*alpha, *beta, d_LDB, d_LDA, d_LDC, AB_order, *N, *M, *K, d_B, d_A, d_C));
  CUDA_CALLER(cudaDeviceSynchronize());
  gettimeofday(&endtime,0);
  double nsec = 1.0e9 *(double)(endtime.tv_sec - starttime.tv_sec) + 1.0e3 * (double)(endtime.tv_usec - starttime.tv_usec);
  printf("The speed of sgemm kernel: %.2e GFLOPS\n", 2.0*(double)(*M)*(double)(*N)*(double)(*K)/nsec);
  CUDA_CALLER(cudaMemcpy2D(C, *LDC*sizeof(float), d_C, d_ldc, (*M)*sizeof(float), *N, cudaMemcpyDeviceToHost));
  CUDA_CALLER(cudaFree(d_A)); CUDA_CALLER(cudaFree(d_B)); CUDA_CALLER(cudaFree(d_C));
}
