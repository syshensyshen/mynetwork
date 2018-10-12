#ifndef __SYSHEN_CUDA_FUNC_HEADER__
#define __SYSHEN_CUDA_FUNC_HEADER__

#include "tools.h"
#include <cublas_v2.h>

#include <driver_types.h>  // cuda driver types

template <typename Dtype>
class syshen_innerproduct {

public:
	syshen_innerproduct(cublasHandle_t handle_);
	~syshen_innerproduct();
	void SetUp();
	void Forward(Dtype *x, Dtype *y);

	inline void setInputParam(int batch, int channels, int height, int width) {
		batch = batch;
		channels = channels;
		height = height;
		width;
	}

private:
	int batch, channels, height, width;
	bool set_cublas_handle_;
	cublasHandle_t handle_t;
};
template class syshen_innerproduct<float>;


#endif
