#include <iostream>
using namespace std;

#define min(x, y) (x > y ? x : y)
#define N 33 * 1024

#define ThreadPerBlock 256

// smallest multiple of threadsPerBlock that is greater than or equal to N
#define blockPerGrid min(32, (N + ThreadPerBlock - 1) / ThreadPerBlock)

__global__ void Vector_Dot_Product(const float *V1, const float *V2,
                                   float *V3) {
  __shared__ float chache[ThreadPerBlock];

  float temp;

  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  const unsigned int chacheindex = threadIdx.x;

  while (tid < N) {
    temp += V1[tid] * V2[tid];

    tid += blockDim.x * gridDim.x;
  }

  chache[chacheindex] = temp;

  __syncthreads();

  int i = blockDim.x / 2;

  while (i != 0) {
    if (chacheindex < i) chache[chacheindex] += chache[chacheindex + i];

    __syncthreads();

    i /= 2;
  }

  if (chacheindex == 0) V3[blockIdx.x] = chache[0];
}

void real_main(void) {
  float *V1_H, *V2_H, *V3_H;
  float *V1_D, *V2_D, *V3_D;

  V1_H = new float[N];
  V2_H = new float[N];
  V3_H = new float[blockPerGrid];

  cudaMalloc((void **)&V1_D, N * sizeof(float));

  cudaMalloc((void **)&V2_D, N * sizeof(float));

  cudaMalloc((void **)&V3_D, blockPerGrid * sizeof(float));

  for (int i = 0; i < N; i++) {
    V1_H[i] = i;

    V2_H[i] = i * 2;
  }

  cudaMemcpy(V1_D, V1_H, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(V2_D, V2_H, N * sizeof(float), cudaMemcpyHostToDevice);

  Vector_Dot_Product<<<blockPerGrid, ThreadPerBlock>>>(V1_D, V2_D, V3_D);

  cudaMemcpy(V3_H, V3_D, N * sizeof(float), cudaMemcpyDeviceToHost);

  cout << "\n Vector Dot Prodcut is : ";

  float sum = 0;

  for (int i = 0; i < blockPerGrid; i++) sum += V3_H[i];
  cout << sum << endl;

  cudaFree(V1_D);
  cudaFree(V2_D);
  cudaFree(V3_D);

  delete[] V1_H;
  delete[] V2_H;
  delete[] V3_H;
}