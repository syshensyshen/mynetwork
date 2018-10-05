#ifndef __SYSHEN_CUDNN_FUNC_HEADER__
#define __SYSHEN_CUDNN_FUNC_HEADER__

#include "tools.h"

template <typename Dtype>
class syshen_convolution {
public:
	syshen_convolution(cudnnHandle_t handle_);
	~syshen_convolution();
	void SetUp();

	inline void setInputParam(int batch, int channels, int height, int width) {
		batch = batch;
		in_channels = channels;
		in_h = height;
		in_w = width;
	}

	inline void setInputKernelParam(int stride_h, int stride_w, int pad_h, int pad_w,
		int dilation_h, int dilation_w, int kernel_h, int kernel_w) {
		stride_h = stride_h;
		stride_w = stride_w;
		pad_h = pad_h;
		pad_w = pad_w;
		dilation_h = dilation_h;
		dilation_w = dilation_w;
		kernel_h = kernel_h;
		kernel_w = kernel_w;
	}

	inline void setOutputParam(int out_batch, int out_channels, int out_h, int out_w) {
		out_batch = output_batch;
		out_channels = out_channels;
		out_h = out_h;
		out_w = out_w;
	}

	void Forward(Dtype *input, Dtype *output, Dtype *weights, Dtype *bias_weights);

protected:

	inline void setStride(int stride_h, int stride_w) {
		stride_h = stride_h;
		stride_w = stride_w;
	}
	inline void setPad(int pad_h, int pad_w) {
		pad_h = pad_h;
		pad_w = pad_w;
	}
	inline void setDliation(int dilation_h, int dilation_w) {
		dilation_h = dilation_h;
		dilation_w = dilation_w;
	}

	inline void setKernel(int kernel_h, int jernel_w) {
		kernel_h = kernel_h;
		kernel_w = kernel_w;
	}
private:
	cudnnTensorDescriptor_t input_desc, output_desc, filter_dsec, conv_desc, bias;
	cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	cudnnHandle_t handle_t;
	cudaStream_t stream;
	cudaEvent_t start;
	size_t workSpaceSize;
	bool use_stream, has_bias, set_cudnn_handle;
	void* workSpace;
	int stride_h, stride_w, pad_h, pad_w;
	int dilation_h, dilation_w, kernel_h, kernel_w;
	int batch, in_channels, in_h, in_w;
	int out_batch, out_channels, out_h, out_w;
};

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

template <typename Dtype>
class syshen_activation {

public:
	syshen_activation(cudnnHandle_t handle_);
	~syshen_activation();
	void SetUp();
	void Forward(Dtype *x, Dtype *y); 
	inline void setInputParam(int batch, int channels, int height, int width) {
		batch = batch;
		channles = channels;
		height = height;
		width = width;
	}
	inline void setMode(cudnnActivationMode_t  mode) {
		mode_ = mode;
	}
private:
	cudnnHandle_t handle_t;
	cudnnActivationDescriptor_t input_desc, output_desc;
	cudnnActivationDescriptor_t act_desc;
	cudnnActivationMode_t  mode_;
	bool set_cudnn_handle;

	int batch, channles, height, width;
};

template <typename Dtype>
class syshen_batchnorm {

public:
	syshen_batchnorm(cudnnHandle_t handle_);
	~syshen_batchnorm();
	void SetUp();
	void Forward(Dtype *x, Dtype *y, Dtype *global_mean, 
		Dtype *global_var, Dtype *bnScale, Dtype *bnBias);

	inline void setInputParam(int batch, int channels, int height, int width) {
		batch = batch;
		channles = channels;
		height = height;
		width;
	}

private:
	cudnnTensorDescriptor_t input_desc, output_desc, scale_bias_desc;
	cudnnBatchNormMode_t mode_;
	cudnnHandle_t handle_t;
	Dtype *input, output;
	bool set_cudnn_handle;
	int batch, channles, height, width;

};


template <typename Dtype>
class syshen_lrn {
	
public:
	syshen_lrn(cudnnHandle_t handle_);
	~syshen_lrn();
	void SetUp();
	void Forward(Dtype *x, Dtype *y);

	inline void setInputParam(int batch, int channels, int height, int width) {
		batch = batch;
		channles = channels;
		height = height;
		width;
	}

	inline void SetLrnParam(unsigned lrnN, double lrnAlpha, double lrnBeta, double lrn) {
		lrnN = lrnN;
		lrnAlpha = lrnAlpha;
		lrnBeta = lrnBeta;
		lrn = lrn;
	}

private:
	cudnnTensorDescriptor_t input_desc, output_desc;
	cudnnLRNDescriptor_t lrn_desc;
	cudnnHandle_t handle_t;
	cudnnLRNMode_t mode_;
	unsigned lrnN;
	double lrnAlpha, lrnBeta, lrn;
	bool set_cudnn_handle;
	int batch, channles, height, width;
};

template <typename Dtype>
class syshen_softmax {

public:
	syshen_softmax(cudnnHandle_t handle_);
	~syshen_softmax();
	void SetUp();
	void Forward(Dtype *x, Dtype *y);

	inline void setInputParam(int batch, int channels, int height, int width) {
		batch = batch;
		channles = channels;
		height = height;
		width;
	}

private:
	cudnnTensorDescriptor_t input_desc, output_desc;
	cudnnSoftmaxAlgorithm_t algo;
	cudnnSoftmaxMode_t mode_;
	cudnnHandle_t handle_t;
	bool set_cudnn_handle;
	int batch, channles, height, width;
};

#endif

