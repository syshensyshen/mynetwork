#include "cudnn_func.hpp"

template <typename Dtype>
syshen_batchnorm<Dtype>::syshen_batchnorm(cudnnHandle_t handle_) {
	CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&input_desc));
	CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&output_desc));
	CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&scale_bias_desc));
	if (handle_) {
		CHECK_CUDNN_ERROR(cudnnCreate(&handle_t));
		set_cudnn_handle = true;
	}
	else {
		handle_t = handle_;
		set_cudnn_handle = false;
	}
}

template <typename Dtype>
syshen_batchnorm<Dtype>::~syshen_batchnorm() {
	CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(input_desc));
	CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(output_desc));
	CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(scale_bias_desc));
	if (set_cudnn_handle) {
		CHECK_CUDNN_ERROR(cudnnDestroy(handle_t));
	}
}

template <typename Dtype>
void syshen_batchnorm<Dtype>::SetUp() {
	CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, 
		CUDNN_DATA_FLOAT, batch, channles, height, width));
	CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, 
		CUDNN_DATA_FLOAT, batch, channles, height, width));
	CHECK_CUDNN_ERROR(cudnnDeriveBNTensorDescriptor(scale_bias_desc, input_desc, mode_));
	CHECK_CUDNN_ERROR(cudnnDeriveBNTensorDescriptor(scale_bias_desc, output_desc, mode_));
	mode_ = CUDNN_BATCHNORM_SPATIAL;
}

template <typename Dtype>
void syshen_batchnorm<Dtype>::Forward(Dtype *x, Dtype *y, Dtype *global_mean,
	Dtype *global_var, Dtype *bnScale, Dtype *bnBias) {
	Dtype alpha = Dtype(1.0);
	Dtype beta = Dype(0.0);
	CHECK_CUDNN_ERROR(cudnnBatchNormalizationForwardInference(handle_t,
		mode_,
		&alpha, 
		&beta, 
		input_desc,
		x, 
		output_desc,
		y,
		scale_bias_desc,
		bnScale,
		bnBias,
		global_mean,
		global_var,
		CUDNN_BN_MIN_EPSILON));
}