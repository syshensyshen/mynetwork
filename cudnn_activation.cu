#include "cudnn_func.hpp"

template <typename Dtype>
syshen_activation<Dtype>::syshen_activation(cudnnHandle_t handle_) {
	CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&input_desc));
	CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&output_desc));
	CHECK_CUDNN_ERROR(cudnnCreateActivationDescriptor(&act_desc));
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
syshen_activation<Dtype>::~syshen_activation() {
	CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(&input_desc));
	CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(&output_desc));
	CHECK_CUDNN_ERROR(cudnnDestroyActivationDescriptor(&act_desc));
	if (set_cudnn_handle_t) {
		CHECK_CUDNN_ERROR(cudnnDestroy(&handle_t));
	}
}

template <typename Dtype>
void syshen_activation<Dtype>::SetUp() {
	CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT, batch, channles, height, width));
	CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT, batch, channles, height, width));
	CHECK_CUDNN_ERROR(cudnnSetActivationDescriptor(act_desc, mode_, CUDNN_NOT_PROPAGATE_NAN, Dtype(0)));
}

template <typename Dtype>
void syshen_activation<Dtype>::Forward(Dtype *x, Dtype *y) {
	Dtype alpha = 1.0f;
	Dtype beta = 0;
	CHECK_CUDNN_ERROR(cudnnActivationForward(handle_t, act_desc,
		&alpha, input_desc, x, &beta, output_desc, y));
}