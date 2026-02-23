#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <vector>

#define N 10000

using namespace std;

__global__ void addition(int* A, int* B, int* C, long long size) {
    long long workIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if(workIdx < size) {
        C[workIdx] = A[workIdx] + B[workIdx];
    }
}

int main() {
    
    int* A;
    int* B;
    int* C;

    int* devA;
    int* devB;
    int* devC;
    
    cudaMallocHost(&A, sizeof(int) * N * N);
    cudaMallocHost(&B, sizeof(int) * N * N);
    cudaMallocHost(&C, sizeof(int) * N * N);

    cudaMalloc(&devA, sizeof(int) * N * N);
    cudaMalloc(&devB, sizeof(int) * N * N);
    cudaMalloc(&devC, sizeof(int) * N * N);

    for (long long i = 0; i < N * N; i++) {
        A[i] = i;
    }

        for (long long i = 0; i < N * N; i++) {
        B[i] = i;
    }

    // 이렇게 하면 cache thrasing 발생함, 
    // 배열 2개 왔다갔다하면서 하면 cache관리 개떡 같이 됨
    // for(long long i = 0; i < N * N; i++) {
    //     A[i] = B[i] = i;
    // }


    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStream_t stream3;
    cudaStream_t stream4;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    cudaEvent_t start;
    cudaEvent_t end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaStream_t streams[] = {stream1, stream2, stream3, stream4};
    long long perStream = N*N/size(streams);

    cudaEventRecord(start, 0);
    for(int i = 0; i < size(streams); i++) {
        long long offset = i * perStream;

        cudaMemcpyAsync(devA + offset, A + offset, sizeof(int)*perStream, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(devB + offset, B + offset, sizeof(int)*perStream, cudaMemcpyHostToDevice, streams[i]);

        int threads = 256;
        int blocks = (perStream + threads-1) / threads;
        addition<<<blocks, threads, 0, streams[i]>>>(devA + offset, devB + offset, devC + offset, perStream);

        cudaMemcpyAsync(C + offset, devC + offset, sizeof(int) * perStream, cudaMemcpyDeviceToHost, streams[i]);
    }

    for(int i = 0; i < size(streams); i++) {
        cudaStreamSynchronize(streams[i]);
    }
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, end);
    cout << "elapsedTime: " << elapsedTime << "ms \n";

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    for(int i = 0; i < size(streams); i++) {
        cudaStreamDestroy(streams[i]);
    }
}