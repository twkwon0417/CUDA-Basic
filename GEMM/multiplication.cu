#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <cstring>

#define N 100

using namespace std;

__global__ void multiplication(int* A, int* B, int* C) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            for(int k = 0; k < N; k++) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }
}

void cpu_multiplication(int* A, int* B, int* C) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            for(int k = 0; k < N; k++) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }
}

int main() {
    int* A;
    int* B;
    int* C;

    cudaMallocHost((void**)&A, sizeof(int)*N*N);
    cudaMallocHost((void**)&B, sizeof(int)*N*N);
    cudaMallocHost((void**)&C, sizeof(int)*N*N);

    for(int i = 0; i < N*N; i++) {
        A[i] = B[i] = i;
    } 
    memset(C, 0, sizeof(int) * N * N);

    int blocks = (N*N+32-1) / 32;
    
    multiplication<<<128,32>>>(A, B, C);

    cudaDeviceSynchronize();

    for(int i=0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            cout << C[i*N + j] << " ";
        }
        cout << "\n";
    }
}