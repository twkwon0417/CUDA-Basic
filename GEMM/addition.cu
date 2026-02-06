#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <vector>

#define N 100
#define M 100

using namespace std;

__global__ void addition(int *A, int *B, int *C) {
  for (int i = 0; i < N * M; i++) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  int *A;
  cudaMallocHost((void **)&A, sizeof(int) * N * M);

  for (int i = 0; i < N * M; i++) {
    A[i] = i;
  }

  int *B;
  cudaMallocHost((void **)&B, sizeof(int) * N * M);
  for (int i = 0; i < N * M; i++) {
    B[i] = i;
  }

  int *C;
  cudaMallocHost((void **)&C, sizeof(int) * N * M);


  /* 요부분 block, thread 갯수 어느 방법으로 설정하는지 잘 알아봐야겠다.
  <<<A, B, C, D>>>
  A: Grid에 Block 몇개 만들어
  B: Block에 Thread 몇개 만들어
  */
  int blocks = (N * M + 32 - 1) / 32;
  addition<<<128, 32>>>(A, B, C);

  cudaDeviceSynchronize();

  int expected = 0;
  int returned = 0;

  for (int i = 0; i < M * N; i++) {
    expected += (A[i] * 2);
    returned += C[i];
  }

  cout << "expected: " << expected << "\n";
  cout << "returned: " << returned << "\n";

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}