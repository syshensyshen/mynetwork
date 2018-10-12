#include<process.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "cudnn_func.hpp"




int main(int argc, char **argv) {

	cudnnHandle_t handle_t;
	CHECK_CUDNN_ERROR(cudnnCreate(&handle_t));

	syshen_convolution<float> conv1(handle_t);
	syshen_pooling<float> pool1(handle_t);
	syshen_convolution<float> conv2(handle_t);
	syshen_pooling<float> pool2(handle_t);
	syshen_activation<float> relu1(handle_t);
	syshen_activation<float> relu2(handle_t);
	syshen_softmax<float> prob(handle_t);

	conv1.setInputKernelParam(1, 1, 0, 0, 0, 0, 5, 5);
	conv1.setInputParam(1, 3, 28, 28);
	conv1.setOutputParam(1, 20, 24, 24);

	pool1.setInputKernelParam(2, 2, 0, 0, 2, 2);
	pool1.setInputParam(1, 20, 24, 24);
	
	conv2.setInputKernelParam(1, 1, 0, 0, 0, 0, 5, 5);
	conv2.setInputParam(1, 20, 12, 12);
	conv2.setOutputParam(1, 50, 8, 8);

	pool2.setInputKernelParam(2, 2, 0, 0, 2, 2);
	pool2.setInputParam(1, 50, 8, 8);

	prob.setInputParam(1, 10, 1, 1);



	system("pause");
	return 0;
}