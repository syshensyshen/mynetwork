#include "cudnn_func.hpp"

template <typename Dtype>
syshen_deconvolution<Dtype>::syshen_deconvolution(cudnnHandle_t handle_) {
	CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&input_desc));
	CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&output_desc));
	CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(&filter_desc));
	CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&conv_desc));
	if (has_bias) {
		CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&bias));
	}
	if (!handle_) {
		CHECK_CUDNN_ERROR(cudnnCreate(&handle_t));
		set_cudnn_handle = true;
	}
	else {
		handle_t = handle_;
		set_cudnn_handle = false;
	}

	if (use_stream) {
		CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
		CHECK_CUDA_ERROR(cudaEventCreate(&start));
	}
	batch = 1;
	in_channels = 1;
	stride_h = 1;
	stride_w = 1;
	pad_h = 1;
	pad_w = 1;
	dilation_h = 1;
	dilation_w = 1;
	kernel_h = 1;
	kernel_w = 1;
}

template <typename Dtype>
syshen_deconvolution<Dtype>::~syshen_deconvolution() {
	CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(input_desc));
	CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(output_desc));
	CHECK_CUDNN_ERROR(cudnnDestroyFilterDescriptor(filter_desc));
	CHECK_CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));
	if (has_bias) {
		CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(bias));
	}
	if (set_cudnn_handle) {
		CHECK_CUDNN_ERROR(cudnnDestroy(handle_t));
	}

	if (use_stream) {
		CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
		//CHECK_CUDA_ERROR(cudaEventDestroy(strat));
	}
}

template<typename Dtype>
void syshen_deconvolution<Dtype>::SetUp() {
	int nStride = in_channels * in_h * in_w;
	int cStride = in_h * in_w;

	CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptorEx(
		input_desc,
		cudnnDataType_t::CUDNN_DATA_FLOAT,
		batch,
		in_channels,
		in_h, in_w, nStride, cStride, in_w, 1));

	CHECK_CUDNN_ERROR(cudnnSetFilter4dDescriptor(
		filter_desc,
		cudnnDataType_t::CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NCHW,
		out_channels, in_channels, kernel_h, kernel_w));

	CHECK_CUDNN_ERROR(cudnnSetConvolution2dDescriptor(
		conv_desc, pad_h, pad_w, stride_h,
		stride_w, dilation_h, dilation_w,
		CUDNN_CROSS_CORRELATION, cudnnDataType_t::CUDNN_DATA_FLOAT));

	/*CHECK_CUDNN_ERROR(cudnnGetConvolution2dForwardOutputDim(
	conv_desc, input_desc, filter_dsec,
	&out_batch, &out_channels, &out_h, &out_w));*/

	CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(
		output_desc, CUDNN_TENSOR_NCHW,
		cudnnDataType_t::CUDNN_DATA_FLOAT,
		out_batch, out_channels, out_h, out_w));

	/*CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterAlgorithm(
		handle_t, input_desc, filter_desc,
		conv_desc, output_desc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
		0, &algo));

	CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(
		handle_t, input_desc, output_desc,
		conv_desc, filter_desc, algo, &workSpaceSize));*/
	CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardDataAlgorithm(
		handle_t, filter_desc, input_desc, conv_desc,
		output_desc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo));
	CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(
		handle_t, filter_desc, input_desc, conv_desc, 
		output_desc, algo, &workSpaceSize));
	if (0 != workSpaceSize)
		CHECK_CUDA_ERROR(cudaMalloc((void**)&workSpace, workSpaceSize));

	if (has_bias) {
		cudnnSetTensor4dDescriptor(bias, CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, out_batch, out_channels, 1, 1);
	}
}

template<typename Dtype>
void syshen_deconvolution<Dtype>::Forward(Dtype *input, Dtype *output, Dtype *weights, Dtype *bias_weights) {
	Dtype conv_alpha = 1.0f;
	Dtype conv_beta = 0;
	CHECK_CUDNN_ERROR(cudnnConvolutionBackwardData(
		handle_t, &conv_alpha, filter_desc, weights, 
		input_desc, input, conv_desc, algo, workSpace,
		workSpaceSize, &conv_beta, output_desc, output));
	if (has_bias) {
		cudnnAddTensor(handle_t, &conv_alpha, bias, bias_weights, &conv_alpha, output_desc, output);
	}
}