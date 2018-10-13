#ifndef __SYSHEN_CUDA_FUNC_HEADER__
#define __SYSHEN_CUDA_FUNC_HEADER__

#include "tools.h"
#include <cublas_v2.h>

#include <driver_types.h>  // cuda driver types


const char* cublasGetErrorString(cublasStatus_t error) {
	switch (error) {
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
	case CUBLAS_STATUS_NOT_SUPPORTED:
		return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
	case CUBLAS_STATUS_LICENSE_ERROR:
		return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
	}
	return "Unknown cublas status";
}


#define CHECK_CUBLAS_ERROR(argument_t)                      \
 cublasStatus_t error_t = argument_t;                       \
 if (error_t != CUBLAS_STATUS_SUCCESS) {                                  \
        printf("Error: %s: %d, ", __FILE__, __LINE__);         \
        printf("code: %d, reason: %s\r\n", error_t, cublasGetErrorString(error_t)); \
        exit(1);                                                 \
}

template <typename Dtype>
class syshen_innerproduct {

public:
	syshen_innerproduct(cublasHandle_t handle_);
	~syshen_innerproduct();
	void SetUp();
	void syshen_innerproduct<Dtype>::Forward(Dtype *x, Dtype *y, Dtype *weight, Dtype *bias);

	inline void setInputParam(int batch, int channels, int height, int width) {
		batch = batch;
		channels = channels;
		height = height;
		width;
	}

private:
	int batch, channels, height, width;
	int output_channels;
	bool set_cublas_handle_;
	bool use_stream, has_bias;
	Dtype *bias_ones;
	cublasHandle_t cublas_handle_t;
};
template class syshen_innerproduct<float>;


template <typename Dtype>
void caffe_gpu_gemm(cublasHandle_t cublas_handle_t, const cublasOperation_t TransA,
	const cublasOperation_t TransB, const int M, const int N, const int K,
	const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
	Dtype* C);


#endif
