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
void __global__ sgemm_kernel(const float alpha, const float beta, const int LDC, const int AB_order,
 const int M, const int N, const int K, cudaTextureObject_t texA, cudaTextureObject_t texB, float *C){
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
  const int k_major_mn_biase = threadIdx.y*blockDim_X;
  const int mn_major_mn_biase = k_major_mn_biase + threadIdx.x;
  float * const share_a_write = Rowmajor_A ? (shared_a+threadIdx.x*share_ld_dim+k_major_mn_biase) : (shared_a+mn_major_mn_biase);
  float * const share_b_write = Rowmajor_B ? (shared_b+threadIdx.y/2*(blockDim_X*2)+threadIdx.y%2+(threadIdx.x<<1)) : (shared_b+threadIdx.x*share_ld_dim+threadIdx.y%2+threadIdx.y/2*(blockDim_X*2));
  const int shared_a_wrptr_update = Rowmajor_A ? 1 : share_ld_dim;
  const int shared_b_wrptr_update = Rowmajor_B ? share_ld_dim : 2;
  int x_pos, y_pos;
  for(int k_pos=0; k_pos<K; k_pos+=blockDim_X){
    x_pos = Rowmajor_A ? k_pos+threadIdx.x : m_base + mn_major_mn_biase;
    y_pos = Rowmajor_A ? m_base+k_major_mn_biase : k_pos;
#pragma unroll
    for(int ll=0; ll<blockDim_X; ll++) a2[ll].x = tex2D<float>(texA, x_pos, y_pos+ll);
    x_pos = Rowmajor_B ? n_base+mn_major_mn_biase : k_pos+threadIdx.x;
    y_pos = Rowmajor_B ? k_pos : n_base+k_major_mn_biase;
#pragma unroll
    for(int ll=0; ll<blockDim_X; ll++) a2[ll].y = tex2D<float>(texB, x_pos, y_pos+ll);
#pragma unroll
    for(int ll=0; ll<blockDim_X; ll++) share_a_write[ll*shared_a_wrptr_update] = a2[ll].x;
#pragma unroll
    for(int ll=0; ll<blockDim_X; ll++) share_b_write[ll*shared_b_wrptr_update] = a2[ll].y;
    __syncthreads();
#pragma unroll 1
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
  CUDA_CALLER(cudaSetDevice(DEVICE_NO)); size_t d_lda, d_ldb, d_ldc; int d_LDC;
  float *d_A, *d_B, *d_C; struct timeval starttime, endtime;
  int A_width, A_height, B_width, B_height;
  if(*transa == 'N'){ A_width = *M; A_height = *K;} else{ A_width = *K; A_height = *M;}
  if(*transb == 'N'){ B_width = *K; B_height = *N;} else{ B_width = *N; B_height = *K;}
  CUDA_CALLER(cudaMallocPitch((void **)&d_A, &d_lda, A_width * sizeof(float), A_height));
  CUDA_CALLER(cudaMallocPitch((void **)&d_B, &d_ldb, B_width * sizeof(float), B_height));
  CUDA_CALLER(cudaMallocPitch((void **)&d_C, &d_ldc, (*M) * sizeof(float), *N));
  d_LDC = d_ldc/sizeof(float);
  CUDA_CALLER(cudaMemcpy2D(d_A, d_lda, A, *LDA*sizeof(float), A_width*sizeof(float), A_height, cudaMemcpyHostToDevice));
  CUDA_CALLER(cudaMemcpy2D(d_B, d_ldb, B, *LDB*sizeof(float), B_width*sizeof(float), B_height, cudaMemcpyHostToDevice));
  CUDA_CALLER(cudaMemcpy2D(d_C, d_ldc, C, *LDC*sizeof(float), *M*sizeof(float), *N, cudaMemcpyHostToDevice));

  struct cudaResourceDesc resDescA, resDescB;
  memset(&resDescA,0,sizeof(resDescA)); memset(&resDescB,0,sizeof(resDescB));
  resDescA.resType = resDescB.resType = cudaResourceTypePitch2D;
  resDescA.res.pitch2D.devPtr = d_A; resDescB.res.pitch2D.devPtr = d_B;
  resDescA.res.pitch2D.width = A_width; resDescB.res.pitch2D.width = B_width;
  resDescA.res.pitch2D.height = A_height; resDescB.res.pitch2D.height = B_height;
  resDescA.res.pitch2D.desc = resDescB.res.pitch2D.desc = cudaCreateChannelDesc<float>();
  resDescA.res.pitch2D.pitchInBytes = d_lda; resDescB.res.pitch2D.pitchInBytes = d_ldb;
  struct cudaTextureDesc texDesc;
  memset(&texDesc,0,sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;
  texDesc.addressMode[0] = texDesc.addressMode[1] = cudaAddressModeBorder;
  texDesc.normalizedCoords = false;
  texDesc.filterMode = cudaFilterModePoint;
  cudaTextureObject_t texA, texB;
  CUDA_CALLER(cudaCreateTextureObject(&texA, &resDescA, &texDesc, NULL));
  CUDA_CALLER(cudaCreateTextureObject(&texB, &resDescB, &texDesc, NULL));

  dim3 gemm_grid((*M-1)/blksize+1, (*N-1)/blksize+1), gemm_block(blockDim_X, blockDim_Y);
  int AB_order = 0; if(*transa == 'N') AB_order |= 2; if(*transb == 'N') AB_order |= 1;
  gettimeofday(&starttime,0);
  CUDA_KERNEL_CALLER(sgemm_kernel<<<gemm_grid, gemm_block>>>(*alpha, *beta, d_LDC, AB_order, *N, *M, *K, texB, texA, d_C));
  CUDA_CALLER(cudaDeviceSynchronize());
  gettimeofday(&endtime,0);
  double nsec = 1.0e9 *(double)(endtime.tv_sec - starttime.tv_sec) + 1.0e3 * (double)(endtime.tv_usec - starttime.tv_usec);
  printf("The speed of sgemm kernel: %.2e GFLOPS\n", 2.0*(double)(*M)*(double)(*N)*(double)(*K)/nsec);
  CUDA_CALLER(cudaDestroyTextureObject(texA)); CUDA_CALLER(cudaDestroyTextureObject(texB));
  CUDA_CALLER(cudaMemcpy2D(C, *LDC*sizeof(float), d_C, d_ldc, (*M)*sizeof(float), *N, cudaMemcpyDeviceToHost));
  CUDA_CALLER(cudaFree(d_A)); CUDA_CALLER(cudaFree(d_B)); CUDA_CALLER(cudaFree(d_C));
}
