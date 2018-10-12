#include "cuda_func.hpp"

template <typename Dtype>
syshen_innerproduct<Dtype>::syshen_innerproduct(cublasHandle_t handle_) {
	set_cublas_handle_ = false;
	if (!handle_) {
		cublasCreate(&handle_t);
		set_cublas_handle_ = true;
	}
	else {
		handle_t = handle_;
		set_cublas_handle_ = false;
	}
}

template <typename Dtype>
syshen_innerproduct<Dtype>::~syshen_innerproduct() {
	if (set_cublas_handle_) {
		cublasDestroy(handle_t);
	}
}

template <typename Dtype>
void syshen_innerproduct<Dtype>::SetUp() {

}

template <typename Dtype>
void syshen_innerproduct<Dtype>::Forward(Dtype *x, Dtype *y) {

}