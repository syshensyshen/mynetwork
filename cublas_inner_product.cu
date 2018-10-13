#include "cuda_func.hpp"
#include <cublas.h>

template <>
void gpu_gemm<float>(cublasHandle_t cublas_handle_t, const cublasOperation_t TransA,
	const cublasOperation_t TransB, const int M, const int N, const int K,
	const float alpha, const float* A, const float* B, const float beta,
	float* C) {
	// Note that cublas follows fortran order.
	int lda = (TransA == CUBLAS_OP_N) ? K : M;
	int ldb = (TransB == CUBLAS_OP_N) ? N : K;

	CHECK_CUBLAS_ERROR(cublasSgemm(cublas_handle_t, TransB, TransA,
		N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void gpu_gemm<double>(cublasHandle_t cublas_handle_t, const cublasOperation_t TransA,
	const cublasOperation_t TransB, const int M, const int N, const int K,
	const double alpha, const double* A, const double* B, const double beta,
	double* C) {
	// Note that cublas follows fortran order.
	int lda = (TransA == CUBLAS_OP_N) ? K : M;
	int ldb = (TransB == CUBLAS_OP_N) ? N : K;

	CHECK_CUBLAS_ERROR(cublasDgemm(cublas_handle_t, TransB, TransA,
		N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <typename Dtype>
syshen_innerproduct<Dtype>::syshen_innerproduct(cublasHandle_t handle_) {
	set_cublas_handle_ = false;
	if (!handle_) {
		cublasCreate(&cublas_handle_t);
		set_cublas_handle_ = true;
	}
	else {
		cublas_handle_t = handle_;
		set_cublas_handle_ = false;
	}
}

template <typename Dtype>
syshen_innerproduct<Dtype>::~syshen_innerproduct() {
	if (set_cublas_handle_) {
		cublasDestroy(cublas_handle_t);
	}
}

template <typename Dtype>
void syshen_innerproduct<Dtype>::SetUp() {

}

template <typename Dtype>
void syshen_innerproduct<Dtype>::Forward(Dtype *x, Dtype *y, Dtype *weight, Dtype *bias) {
	Dtype alpha = 1.0;
	Dtype beta = 0.0;
	gpu_gemm<Dtype>(
		cublas_handle_t, CUBLAS_OP_N, CUBLAS_OP_T,
		batch, output_channels, channels, alpha,
		x, weight, beta, y);
	if (has_bias) {
		beta = 1.0;
		gpu_gemm<Dtype>(
			cublas_handle_t, CUBLAS_OP_N, CUBLAS_OP_N,
			batch, output_channels, 1, alpha,
			bias_ones, bias, beta, y);
	}

}