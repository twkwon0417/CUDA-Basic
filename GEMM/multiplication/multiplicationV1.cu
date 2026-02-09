#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <cstring>

// 행렬 크기 (너무 크면 출력할 때 보기 힘드므로 100 유지)
#define N 100

using namespace std;

// CUDA 에러 체크 매크로 (필수)
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// __global__: GPU에서 실행되는 커널
// 자료형을 long long으로 변경
__global__ void multiplication(long long* A, long long* B, long long* C) {
    // 2D 인덱싱 계산
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        long long temp = 0; // 누적 변수도 반드시 long long이어야 함
        for(int k = 0; k < N; k++) {
            // A와 B의 요소가 크기 때문에 곱셈 결과는 매우 커짐
            temp += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = temp;
    }
}

int main() {
    long long* A;
    long long* B;
    long long* C;

    size_t matrixSize = sizeof(long long) * N * N;

    // Unified Memory (cudaMallocHost) 할당 및 에러 체크
    // sizeof(long long) 사용 주의
    cudaCheckError(cudaMallocHost((void**)&A, matrixSize));
    cudaCheckError(cudaMallocHost((void**)&B, matrixSize));
    cudaCheckError(cudaMallocHost((void**)&C, matrixSize));

    // 데이터 초기화: 큰 값을 넣어 int 범위를 넘기도록 유도
    for(int i = 0; i < N*N; i++) {
        // 예: 1,000,000 이상의 값. 
        // 곱하면 10^12(1조) 단위가 되어 일반 int(약 20억)로는 오버플로우 발생
        A[i] = 1000000LL + i; 
        B[i] = 1000000LL + i;
    } 
    memset(C, 0, matrixSize);

    // 실행 설정: 2D 그리드
    dim3 threadsPerBlock(16, 16); // 256 threads
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    printf("Matrix Size: %d x %d\n", N, N);
    printf("Computing with long long...\n");

    // 커널 실행
    multiplication<<<blocksPerGrid, threadsPerBlock>>>(A, B, C);

    // 커널 런칭 에러 체크
    cudaCheckError(cudaPeekAtLastError());
    // 디바이스 동기화 (GPU 작업 완료 대기)
    cudaCheckError(cudaDeviceSynchronize());

    // 결과 확인 (일부만 출력)
    cout.precision(20); // 큰 수 출력을 위해 정밀도 조정
    cout << "\n--- Result Sample (Corner elements) ---" << endl;
    cout << "C[0][0]       : " << C[0] << endl;
    cout << "C[N-1][N-1]   : " << C[N*N - 1] << endl;

    // 전체 출력은 너무 길 수 있으므로 주석 처리하거나 필요 시 해제
    /*
    for(int i=0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            cout << C[i*N + j] << " ";
        }
        cout << "\n";
    }
    */

    // 메모리 해제-skrr
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}