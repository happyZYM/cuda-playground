#include <cassert>
#include <chrono>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <random>

// You may increase this value to test larger matrices
// But it will be slow on CPU
constexpr int MAXN = 8192;

/**
 * @brief A naive implementation of matrix multiplication on CPU.
 * Perform C = A * B, where A is M x K, B is K x N, and C is M x N.
 */
void naiveSgemm(float *a, float *b, float *c, const int M, const int N,
                const int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += a[m * K + k] * b[k * N + n];
      }
      c[m * N + n] = sum;
    }
  }
}

/**
 * @brief Optimized implementation of matrix multiplication on GPU using shared memory.
 * Perform C = A * B, where A is M x K, B is K x N, and C is M x N.
 */
 template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1) ZYMSgemm2D(const float *a,const float *b, float *c, const int M,
                                     const int N, const int K) {
  const int block_row = blockIdx.x;
  const int block_col = blockIdx.y;
  const int elements_per_block = BM * BN;
  const int threads_per_block = (BM*BN)/(TM*TN);
  assert(blockDim.x == threads_per_block);
  const int thread_row = threadIdx.x / (BN/TN);
  const int thread_col = threadIdx.x % (BN/TN);
  __shared__ float shared_a[BM*BK];
  __shared__ float shared_b[BK*BN];
  a+=block_row*BM*K;
  b+=block_col*BN;
  c+=block_row*BM*N+block_col*BN;
  const int load_a_col = threadIdx.x % BK;
  const int load_a_row = threadIdx.x / BK;
  const int load_a_row_stride = threads_per_block / BK;
  const int load_b_col = threadIdx.x % BN;
  const int load_b_row = threadIdx.x / BN;
  const int load_b_row_stride = threads_per_block / BN;
  float result_cache[TM*TN]={0.0};
  float a_cache[TM]={0.0};
  float b_cache[TN]={0.0};
  for(int k_idx=0;k_idx<K;k_idx+=BK) {
    for(int load_a_offset=0;load_a_offset<BM;load_a_offset+=load_a_row_stride) {
      shared_a[(load_a_offset+load_a_row)*BK+load_a_col]=a[(load_a_offset+load_a_row)*K+load_a_col];
    }
    for(int load_b_offset=0;load_b_offset<BK;load_b_offset+=load_b_row_stride) {
      shared_b[(load_b_offset+load_b_row)*BN+load_b_col]=b[(load_b_offset+load_b_row)*N+load_b_col];
    }
    __syncthreads();
    a+=BK;
    b+=BK*N;
    for(int dot_idx=0;dot_idx<BK;dot_idx++) {
      for(int i=0;i<TM;i++) {
        a_cache[i]=shared_a[(thread_row*TM+i)*BK+dot_idx];
      }
      for(int i=0;i<TN;i++) {
        b_cache[i]=shared_b[dot_idx*BN+thread_col*TN+i];
      }
      for(int i=0;i<TM;i++) {
        for(int j=0;j<TN;j++) {
          result_cache[i*TN+j]+=a_cache[i]*b_cache[j];
        }
      }
    }
    __syncthreads();
  }
  for(int i=0;i<TM;i++) {
    for(int j=0;j<TN;j++) {
      c[(thread_row*TM+i)*N+thread_col*TN+j]=result_cache[i*TN+j];
    }
  }
}

/**
 * @brief Launch ZYMSgemm2D kernel.
 * @details see https://siboehm.com/articles/22/CUDA-MMM
 */
void launchSgemm2D(const float *a,const float *b, float *c, const int M, const int N,
                   const int K) {
  const int BK=8;
  const int TM=8;
  const int TN=8;
  if(M>=128&&N>=128&&K>=128){
    const int BM=128;
    const int BN=128;
    dim3 gridDim((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 blockDim((BM * BN) / (TM * TN));
    ZYMSgemm2D<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(a, b, c, M, N, K);
  }
  else{
    const int BM=64;
    const int BN=64;
    dim3 gridDim((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 blockDim((BM * BN) / (TM * TN));
    ZYMSgemm2D<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(a, b, c, M, N, K);
  }
}

void initialize(float *a, float *b, float *c, const int M, const int N,
                const int K) {
  auto gen = std::mt19937(2024);
  auto dis = std::uniform_real_distribution<float>(-1.0, 1.0);
  for (int i = 0; i < M * K; ++i) {
    a[i] = dis(gen);
  }
  for (int i = 0; i < K * N; ++i) {
    b[i] = dis(gen);
  }
  for (int i = 0; i < M * N; ++i) {
    c[i] = 0.0;
  }
}

/** 
 * @brief Launch sgemm using cuBLAS
 */
void launchCublasSgemm(float *a, float *b, float *c, const int M, const int N,
                       const int K) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0;
  float beta = 0.0;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, N, a, K,
              &beta, c, N);
}


int main() {
  float *a, *b, *c;
  a = new float[MAXN * MAXN];
  b = new float[MAXN * MAXN];
  c = new float[MAXN * MAXN];
  initialize(a, b, c, MAXN, MAXN, MAXN);

  // ********** CPU **********
  auto start = std::chrono::high_resolution_clock::now();
  // naiveSgemm(a, b, c, MAXN, MAXN, MAXN);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  printf("CPU time: %.3fs\n", elapsed.count());

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, MAXN * MAXN * sizeof(float));
  cudaMalloc(&d_b, MAXN * MAXN * sizeof(float));
  cudaMalloc(&d_c, MAXN * MAXN * sizeof(float));
  cudaMemcpy(d_a, a, MAXN * MAXN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, MAXN * MAXN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, MAXN * MAXN * sizeof(float), cudaMemcpyHostToDevice);

  // ********** GPU **********
  start = std::chrono::high_resolution_clock::now();
  launchSgemm2D(d_a, d_b, d_c, MAXN, MAXN, MAXN);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  cudaMemcpy(c, d_c, MAXN * MAXN * sizeof(float), cudaMemcpyDeviceToHost);
  printf("d_c[108873]=%f\n", c[108873]);
  elapsed = end - start;
  printf("GPU time: %.3fs\n", elapsed.count());

  // ********** cuBLAS **********
  start = std::chrono::high_resolution_clock::now();
  launchCublasSgemm(d_a, d_b, d_c, MAXN, MAXN, MAXN);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  cudaMemcpy(c, d_c, MAXN * MAXN * sizeof(float), cudaMemcpyDeviceToHost);
  printf("d_c[108873]=%f\n", c[108873]);
  elapsed = end - start;
  printf("cuBLAS time: %.3fs\n", elapsed.count());
}
