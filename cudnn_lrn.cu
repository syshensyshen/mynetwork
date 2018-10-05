#include "cudnn_func.hpp"

template <typename Dtype>
syshen_lrn<Dtype>::syshen_lrn(cudnnHandle_t handle_) {
	CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&input_desc));
	CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&output_desc));
	CHECK_CUDNN_ERROR(cudnnCreateLRNDescriptor(&lrn_desc));
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
syshen_lrn<Dtype>::~syshen_lrn() {
	CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(input_desc));
	CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(output_desc));
	CHECK_CUDNN_ERROR(cudnnDestroyLRNDescriptor(lrn_desc));
	if (set_cudnn_handle) {
		CHECK_CUDNN_ERROR(cudnnDestroy(handle_t));
	}
}

template <typename Dtype>
void syshen_lrn<Dtype>::SetUp() {
	CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT, batch, channles, height, width));
	CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT, batch, channles, height, width));
	CHECK_CUDNN_ERROR(cudnnSetLRNDescriptor(lrn_desc, lrnN, lrnAlpha, lrnBeta, lrnK));
}

template <typename Dtype>
void syshen_lrn<Dtype>::Forward(Dtype *x, Dtype *y) {
	Dtype alpha = Dtype(1.0);
	Dtype beta = Dtype(0.0);
	cudnnLRNCrossChannelForward(handle_t, lrn_desc, 
		CUDNN_LRN_CROSS_CHANNEL_DIM1, &alpha, input_desc, x, &beta, output_desc, y);
}