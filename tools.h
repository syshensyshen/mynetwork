#ifndef __SYSHEN_CUDA_TEST_HEADER__
#define __SYSHEN_CUDA_TEST_HEADER__
#include<process.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cudnn.h>

#define CHECK_CUDA_ERROR(argument_t) {                           \
    cudaError_t error_t = argument_t;                              \
    if (error_t != cudaSuccess) {                                  \
        printf("Error: %s: %d, ", __FILE__, __LINE__);         \
        printf("code: %d, reason: %s\r\n", error_t, cudaGetErrorString(error_t)); \
        exit(1);                                                 \
	}                                                            \
}

#define CHECK_CUDNN_ERROR(argument_t) {                           \
    cudnnStatus_t error_t = argument_t;                              \
    if (error_t != CUDNN_STATUS_SUCCESS) {                                  \
        printf("Error: %s: %d, ", __FILE__, __LINE__);         \
        printf("code: %d, reason: %s\r\n", error_t, cudnnGetErrorString(error_t)); \
        exit(1);                                                 \
	}                                                            \
}

__global__ void checkIndex();

void test_check();

void test_change_block();




#endif
