//nvcc --shared -Xcompiler -fPIC -Xptxas -O3,--verbose -arch=sm_70 dgemm.cu -o dgemm.so
#ifndef blockDim_X
 #define blockDim_X 16
#endif
#ifndef blockDim_Y
 #define blockDim_Y 4
#endif
#define blksize (blockDim_X*blockDim_Y)
#define share_ld_dim (blksize+1)
void __global__ dgemm_kernel(const double alpha, const double beta, const int LDA, const int LDB, const int LDC, const int AB_order,
 const int M, const int N, const int K, const double * __restrict__ A, const double * __restrict__ B, double *C){
  /* row_major matrix C; blockdim(blockDim_X, blockDim_Y) */
  const int Rowmajor_A = AB_order & 1;
  const int Rowmajor_B = AB_order & 2;
  register double c[blockDim_X][blockDim_Y];
  __shared__ double shared_a[share_ld_dim*blockDim_X], shared_b[share_ld_dim*blockDim_X];
  const int m_base = blockIdx.y * blksize;
  const int n_base = blockIdx.x * blksize;
#pragma unroll
  for(int i=0; i<blockDim_X; i++){
#pragma unroll
    for(int j=0; j<blockDim_Y; j++) c[i][j] = 0.0;
  }
  double *sa = shared_a + threadIdx.y*blockDim_X, *sb = shared_b + threadIdx.x;
  register double a[blockDim_X*2], b[blockDim_Y];
  bool load_cond;
  const int k_major_mn_biase = threadIdx.y*blockDim_X;
  const int mn_major_mn_biase = k_major_mn_biase + threadIdx.x;
  double * const share_a_write = Rowmajor_A ? (shared_a+threadIdx.x*share_ld_dim+k_major_mn_biase) : (shared_a+mn_major_mn_biase);
  double * const share_b_write = Rowmajor_B ? (shared_b+threadIdx.y*blockDim_X+threadIdx.x) : (shared_b+threadIdx.x*share_ld_dim+threadIdx.y*blockDim_X);
  const int a_ptr_inc_last = Rowmajor_A ? (blockDim_X-blockDim_X*LDA) : 0;
  const int b_ptr_inc_last = Rowmajor_B ? 0 : (blockDim_X-blockDim_X*LDB);
  const double * __restrict__ a_ptr = Rowmajor_A ? (A+(m_base+k_major_mn_biase)*LDA+threadIdx.x) : (A+m_base+mn_major_mn_biase);
  const double * __restrict__ b_ptr = Rowmajor_B ? (B+n_base+mn_major_mn_biase) : (B+(n_base+k_major_mn_biase)*LDB+threadIdx.x);
  int load_cond_a_lhs = Rowmajor_A ? (threadIdx.x-K) : (m_base+mn_major_mn_biase-M);
  int load_cond_b_lhs = Rowmajor_B ? (n_base+mn_major_mn_biase-N) : (threadIdx.x-K);
  const int update_load_cond_a_lhs = Rowmajor_A ? blockDim_X : 0;
  const int update_load_cond_b_lhs = Rowmajor_B ? 0 : blockDim_X;
  int l2_cond_a_lhs = Rowmajor_A ? (m_base+k_major_mn_biase-M) : (-K);
  int l2_cond_b_lhs = Rowmajor_B ? (-K) : (n_base+k_major_mn_biase-N);
  const int update_l2_cond_a_lhs = Rowmajor_A ? 0 : blockDim_X;
  const int update_l2_cond_b_lhs = Rowmajor_B ? blockDim_X : 0;
  const int shared_a_wrptr_update = Rowmajor_A ? 1 : share_ld_dim;
  const int shared_b_wrptr_update = Rowmajor_B ? share_ld_dim : 1;
  for(int k_pos=0; k_pos<K; k_pos+=blockDim_X){
    load_cond = (load_cond_a_lhs<0);
    if(load_cond && l2_cond_a_lhs <= -blockDim_X){
#pragma unroll
      for(int ll=0; ll<blockDim_X; ll++){
        a[ll] = *a_ptr; a_ptr += LDA;
      }
    }else{
#pragma unroll
      for(int ll=0; ll<blockDim_X; ll++){
        if(l2_cond_a_lhs+ll<0 && load_cond) a[ll] = *a_ptr;
        else a[ll] = 0.0;
        a_ptr += LDA;
      }
    }
    a_ptr += a_ptr_inc_last;
    load_cond_a_lhs += update_load_cond_a_lhs; l2_cond_a_lhs += update_l2_cond_a_lhs;
    load_cond = (load_cond_b_lhs<0);
    if(load_cond && l2_cond_b_lhs <= -blockDim_X){
#pragma unroll
      for(int ll=0; ll<blockDim_X; ll++){
        a[ll+blockDim_X] = *b_ptr; b_ptr += LDB;
      }
    }else{
#pragma unroll
      for(int ll=0; ll<blockDim_X; ll++){
        if(l2_cond_b_lhs+ll<0 && load_cond) a[ll+blockDim_X] = *b_ptr;
        else a[ll+blockDim_X] = 0.0;
        b_ptr += LDB;
      }
    }
    b_ptr += b_ptr_inc_last;
    load_cond_b_lhs += update_load_cond_b_lhs; l2_cond_b_lhs += update_l2_cond_b_lhs;
#pragma unroll
    for(int ll=0; ll<blockDim_X; ll++) share_a_write[ll*shared_a_wrptr_update] = a[ll];
#pragma unroll
    for(int ll=0; ll<blockDim_X; ll++) share_b_write[ll*shared_b_wrptr_update] = a[ll+blockDim_X];
    __syncthreads();
#pragma unroll 4
    for(int ll=0; ll<blockDim_X; ll++){
#pragma unroll
      for(int i=0; i<blockDim_Y; i++)  b[i] = sb[ll*share_ld_dim+i*blockDim_X];
#pragma unroll
      for(int i=0; i<blockDim_X; i++){
        a[i] = sa[ll*share_ld_dim+i];
#pragma unroll
        for(int j=0; j<blockDim_Y; j++) c[i][j] += a[i] * b[j];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for(int i=0; i<blockDim_X; i++){
#pragma unroll
    for(int j=0; j<blockDim_Y; j++) c[i][j] *= alpha;
  }
  int m_pos = m_base+threadIdx.y*blockDim_X, n_pos = 0;
  double *c_ptr = C + m_pos * LDC;
#pragma unroll
  for(int i=0; i<blockDim_X; i++){
    if(m_pos>=M) return;
    n_pos = n_base+threadIdx.x;
#pragma unroll
    for(int j=0; j<blockDim_Y; j++){
      if(n_pos<N){
        c[i][j] += c_ptr[n_pos] * beta; c_ptr[n_pos] = c[i][j];
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
extern "C" void dgemm_(const char *transa, const char *transb, const int *M, const int *N, const int *K,
 const double *alpha, const double *A, const int *LDA, const double *B, const int *LDB, const double *beta, double *C, const int *LDC){
  CUDA_CALLER(cudaSetDevice(DEVICE_NO)); size_t d_lda, d_ldb, d_ldc; int d_LDA, d_LDB, d_LDC;
  double *d_A, *d_B, *d_C; struct timeval starttime, endtime;
  int A_width, A_height, B_width, B_height;
  if(*transa == 'N'){ A_width = *M; A_height = *K;} else{ A_width = *K; A_height = *M;}
  if(*transb == 'N'){ B_width = *K; B_height = *N;} else{ B_width = *N; B_height = *K;}
#ifdef ALIGNED_ALLOC
  CUDA_CALLER(cudaMallocPitch((void **)&d_A, &d_lda, A_width * sizeof(double), A_height));
  CUDA_CALLER(cudaMallocPitch((void **)&d_B, &d_ldb, B_width * sizeof(double), B_height));
  CUDA_CALLER(cudaMallocPitch((void **)&d_C, &d_ldc, (*M) * sizeof(double), *N));
  d_LDA = d_lda/sizeof(double); d_LDB = d_ldb/sizeof(double); d_LDC = d_ldc/sizeof(double);
#else
  d_LDA = A_width; d_LDB = B_width; d_LDC = *M;
  d_lda = d_LDA * sizeof(double); d_ldb = d_LDB * sizeof(double); d_ldc = d_LDC * sizeof(double);
  CUDA_CALLER(cudaMalloc((void **)&d_A, d_lda  * A_height));
  CUDA_CALLER(cudaMalloc((void **)&d_B, d_ldb  * B_height));
  CUDA_CALLER(cudaMalloc((void **)&d_C, d_ldc  * (*N)));
#endif
  CUDA_CALLER(cudaMemcpy2D(d_A, d_lda, A, *LDA*sizeof(double), A_width*sizeof(double), A_height, cudaMemcpyHostToDevice));
  CUDA_CALLER(cudaMemcpy2D(d_B, d_ldb, B, *LDB*sizeof(double), B_width*sizeof(double), B_height, cudaMemcpyHostToDevice));
  CUDA_CALLER(cudaMemcpy2D(d_C, d_ldc, C, *LDC*sizeof(double), *M*sizeof(double), *N, cudaMemcpyHostToDevice));
  dim3 gemm_grid((*M-1)/blksize+1, (*N-1)/blksize+1), gemm_block(blockDim_X, blockDim_Y);
  int AB_order = 0; if(*transa == 'N') AB_order |= 2; if(*transb == 'N') AB_order |= 1;
  gettimeofday(&starttime,0);
  CUDA_KERNEL_CALLER(dgemm_kernel<<<gemm_grid, gemm_block>>>(*alpha, *beta, d_LDB, d_LDA, d_LDC, AB_order, *N, *M, *K, d_B, d_A, d_C));
  CUDA_CALLER(cudaDeviceSynchronize());
  gettimeofday(&endtime,0);
  double nsec = 1.0e9 *(double)(endtime.tv_sec - starttime.tv_sec) + 1.0e3 * (double)(endtime.tv_usec - starttime.tv_usec);
  printf("The speed of dgemm kernel: %.2e GFLOPS\n", 2.0*(double)(*M)*(double)(*N)*(double)(*K)/nsec);
  CUDA_CALLER(cudaMemcpy2D(C, *LDC*sizeof(double), d_C, d_ldc, (*M)*sizeof(double), *N, cudaMemcpyDeviceToHost));
  CUDA_CALLER(cudaFree(d_A)); CUDA_CALLER(cudaFree(d_B)); CUDA_CALLER(cudaFree(d_C));
}
