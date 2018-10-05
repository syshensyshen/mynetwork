#ifndef __SYSHEN_CUDNN_POOLING_HEADER__
#define __SYSHEN_CUDNN_POOLING_HEADER__

#include "tools.h"

template <typename Dtype>
class syshen_pooling {

public:
	syshen_pooling(cudnnHandle_t handle_);
	~syshen_pooling();
	void Forward(Dtype *x, Dtype *y);
	void SetUp(cudnnPoolingMode_t mode);

	inline void setInputParam(int batch, int channels, int height, int width) {
		batch = batch;
		channels = channels;
		height = height;
		width = width;
	}

	inline void setInputKernelParam(int stride_h, int stride_w, int pad_h, int pad_w, 
		int kernel_h, int kernel_w) {
		stride_h = stride_h;
		stride_w = stride_w;
		pad_h = pad_h;
		pad_w = pad_w;
		kernel_h = kernel_h;
		kernel_w = kernel_w;
	}

private:
	cudnnHandle_t handle_t;
	cudnnPoolingDescriptor_t poolingDesc;
	cudnnTensorDescriptor_t input_desc, output_desc;
	cudnnPoolingMode_t mode_;

	int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w;
	int batch, channels, height, width;
	bool set_cudnn_handle;
};

#endif
