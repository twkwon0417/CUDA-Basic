#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <vector>

#define N 10000

using namespace std;

 void cpu_addition(int *A, int *B, int *C) {
  for (long long i = 0; i < N * N; i++) {
    C[i] = A[i] + B[i];
  }
}

__global__ void addition(int* A, int* B, int* C) {
  long long workIndex = threadIdx.x + blockDim.x * blockIdx.x;

  // CPU에서 Kernel을 호출하는 순간, GPU는 오직 그 커널만을 위한 Grid를 생성한다. 따라서 workIdx는 무조건 0부터 시작이 보장 -> 문제될거 
  if(workIndex < N * N) {
    C[workIndex] = A[workIndex] + B[workIndex];
  }
}

int main() {
  int *A;
  int* devA = nullptr;
  cudaMallocHost((void **)&A, sizeof(int) * N * N);
  cudaMalloc((void **)&devA, sizeof(int) * N * N);

  // init 
  for (long long i = 0; i < N * N; i++) {
    A[i] = i;
  }

  int *B;
  int* devB = nullptr;
  cudaMallocHost((void **)&B, sizeof(int) * N * N);
  cudaMalloc((void **)&devB, sizeof(int) * N * N);

  for (long long i = 0; i < N * N; i++) {
    B[i] = i;
  }

  int *C;
  int* devC = nullptr;  
  cudaMallocHost((void **)&C, sizeof(int) * N * N);
  cudaMalloc((void **)&devC, sizeof(int) * N * N);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* 요부분 block, thread 갯수 어느 방법으로 설정하는지 잘 알아봐야겠다.
  <<<A, B, C, D>>>
  A: Grid에 Block 몇개 만들어
  B: Block에 Thread 몇개 만들어
  */
  int threads = 256;
  int blocks = (N * N + threads - 1) / threads;

  cudaEventRecord(start, stream);

  cudaMemcpy(devA, A, N*N*sizeof(int), cudaMemcpyDefault);
  cudaMemcpy(devB, B, N*N*sizeof(int), cudaMemcpyDefault);

  // arg3: block내에서 사용되는 shared memory의 크기
  addition<<<blocks, threads, 0, stream>>>(devA, devB, devC);

  // cudaDeviceSynchronize();
  cudaStreamSynchronize(stream);

  cudaMemcpy(C, devC, N*N*sizeof(int), cudaMemcpyDeviceToHost);
  // 명시적으로 써있는 아래아 같은 함수 선호
  // cudaMemcpyHostToDevice();

  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cout << "Kernel execution time: " << elapsedTime << " ms" << "\n";

  long long expected = 0;
  long long returned = 0;

  for (long long i = 0; i < N * N; i++) {
    expected += (A[i] * 2);
    returned += C[i];
  }

  cout << "expected: " << expected << "\n";
  cout << "returned: " << returned << "\n";

  cudaFreeHost(A); 
  cudaFreeHost(B);
  cudaFreeHost(C);
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);
}