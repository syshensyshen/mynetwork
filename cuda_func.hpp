#ifndef __SYSHEN_CUDA_FUNC_HEADER__
#define __SYSHEN_CUDA_FUNC_HEADER__

#include "tools.h"

template <typename Dtype>
class syshen_innerproduct {

public:
	explicit syshen_innerproduct(cublasHandle_t handle_);
	~syshen_innerproduct();
	void SetUp();
	void Forward(Dtype *x, Dtype *y, Dtype *weight, Dtype *bias);

	inline void setInputParam(int batch, int channels, int height, int width) {
		batch = batch;
		channels = channels;
		height = height;
		width;
	}

	inline void setOutputParam(int out_channels) {
		output_channels = out_channels;
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
void gpu_gemm(cublasHandle_t cublas_handle_t, const cublasOperation_t TransA,
	const cublasOperation_t TransB, const int M, const int N, const int K,
	const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
	Dtype* C);


#endif
