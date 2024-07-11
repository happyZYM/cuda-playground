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

const int TILE_SIZE = 32;
/**
 * @brief Optimized implementation of matrix multiplication on GPU using shared memory.
 * Perform C = A * B, where A is M x K, B is K x N, and C is M x N.
 */
__global__ void ZYMSgemm2D(float *a, float *b, float *c, const int M,
                                     const int N, const int K) {
  
  // Shared memory for submatrices of A and B
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];
  
  // Calculate row and column index of the element
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  
  // Accumulator for the result
  float value = 0.0f;
  
  // Loop over all the tiles
  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    // Load the tile elements into shared memory
    if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
      As[threadIdx.y][threadIdx.x] = a[row * K + t * TILE_SIZE + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    if (col < N && (t * TILE_SIZE + threadIdx.y) < K) {
      Bs[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * N + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    // Synchronize to make sure the submatrices are loaded
    __syncthreads();
    
    // Multiply the two matrices together
    for (int k = 0; k < TILE_SIZE; ++k) {
      value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    
    // Synchronize to make sure that the computation is done before loading new tiles
    __syncthreads();
  }
  
  // Write the result back to the global memory
  if (row < M && col < N) {
    c[row * N + col] = value;
  }
}

/**
 * @brief Launch ZYMSgemm2D kernel.
 */
void launchSgemm2D(float *a, float *b, float *c, const int M, const int N,
                   const int K) {
  dim3 block(TILE_SIZE, TILE_SIZE); // 256 threads per block (16 * 16 = 256)
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  ZYMSgemm2D<<<grid, block>>>(a, b, c, M, N, K);
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
  printf("d_c[0][0]=%f\n", c[0]);
  elapsed = end - start;
  printf("GPU time: %.3fs\n", elapsed.count());

  // ********** cuBLAS **********
  start = std::chrono::high_resolution_clock::now();
  launchCublasSgemm(d_a, d_b, d_c, MAXN, MAXN, MAXN);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  cudaMemcpy(c, d_c, MAXN * MAXN * sizeof(float), cudaMemcpyDeviceToHost);
  printf("d_c[0][0]=%f\n", c[0]);
  elapsed = end - start;
  printf("cuBLAS time: %.3fs\n", elapsed.count());
}
