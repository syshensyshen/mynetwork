#include "cudnn_func.hpp"

template <typename Dtype>
syshen_softmax<Dtype>::syshen_softmax(cudnnHandle_t handle_) {
	CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&input_desc));
	CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&output_desc));
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
syshen_softmax<Dtype>::~syshen_softmax() {
	CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(&input_desc));
	CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(&output_desc));

	if (set_cudnn_handle_t) {
		CHECK_CUDNN_ERROR(cudnnDestroy(&handle_t));
	}
}

template <typename Dtype>
void syshen_softmax<Dtype>::SetUp() {
	CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT, batch, channles, height, width));
	CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT, batch, channles, height, width));
}

template <typename Dtype>
void syshen_softmax<Dtype>::Forward(Dtype *x, Dtype *y) {
	Dtype alpha = Dtype(1.0);
	Dtype beta = Dtype(0.0);
	mode_ = CUDNN_SOFTMAX_MODE_CHANNEL;
	CHECK_CUDNN_ERROR(cudnnSoftmaxForward(handle_t, CUDNN_SOFTMAX_ACCURATE, mode_,
		&alpha, input_desc, x, &beta, output_desc, y));
}