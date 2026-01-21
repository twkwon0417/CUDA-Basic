#include <stdio.h>
#include <cuda_runtime.h>

// GPU에서 실행될 커널 함수
__global__ void helloCUDA() {
    printf("Hello form GPU thread %d!\n", threadIdx.x);
}

int main() {
    // 1. GPU 장치 개수 확인
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("❌ 실패: 감지된 CUDA 장치가 없습니다.\n");
        return -1;
    }

    // 2. 첫 번째 GPU 정보 가져오기
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int driverVersion = 0;
    cudaDriverGetVersion(&driverVersion);

    printf("✅ 성공! 감지된 GPU: %s\n", prop.name);
    printf(" - Compute Capability: %d.%d\n", prop.major, prop.minor); // 7.5 출력 예정
    printf(" - CUDA Driver Version (Code): %d.%d\n", 
           driverVersion / 1000, (driverVersion % 100) / 10);

    // 3. 커널 실행 (GPU 작업 요청)
    printf("\n[Kernel 실행 테스트]\n");
    helloCUDA<<<1, 5>>>(); // 1개 블록, 5개 스레드
    
    // 4. 동기화 (GPU 작업 끝날 때까지 대기)
    cudaDeviceSynchronize();

    return 0;
}