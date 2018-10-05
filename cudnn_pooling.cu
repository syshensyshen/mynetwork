#include "cudnn_func.hpp"

template <typename Dtype>
syshen_pooling<Dtype>::syshen_pooling(cudnnHandle_t handle_) {
	CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&input_desc));
	CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&output_desc));
	CHECK_CUDNN_ERROR(cudnnCreatePoolingDescriptor(&poolingDesc));
	if (!handle_) {
		CHECK_CUDNN_ERROR(cudnnCreate(&handle_t));
		set_cudnn_handle = true;
	}
	else {
		handle_t = handle_;
		set_cudnn_handle = false;
	}
}

template <typename Dtype>
syshen_pooling<Dtype>::~syshen_pooling() {
	CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(input_desc));
	CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(output_desc));
	CHECK_CUDNN_ERROR(cudnnDestroyPoolingDescriptor(poolingDesc));
	if (set_cudnn_handle) {
		CHECK_CUDNN_ERROR(cudnnDestroy(handle_t));
	}
}

template <typename Dtype>
void syshen_pooling<Dtype>::SetUp(cudnnPoolingMode_t mode) {
	mode_ = mode;
	CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT, batch, channels, height, width));
	CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT, batch, channels, height, width));
	CHECK_CUDNN_ERROR(cudnnSetPooling2dDescriptor(poolingDesc, mode_,
		CUDNN_NOT_PROPAGATE_NAN, kernel_h, kernel_w, 
		pad_h, pad_w, stride_h, stride_w));
}

template <typename Dtype>
void syshen_pooling<Dtype>::Forward(Dtype *x, Dtype *y) {
	Dtype alpha = 1.0f;
	Dtype beta = 0;
	CHECK_CUDNN_ERROR(cudnnPoolingForward(handle_t, poolingDesc,
		&alpha, input_desc, x, &beta, output_desc, y));
}