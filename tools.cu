
#include "tools.h"
__global__ void checkIndex() {
	printf("threadIdx: (%d, %d, %d) blockIdx: (%d, %d, %d) blockDim: (%d, %d, %d) gridDim: (%d, %d, %d)\r\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z);
}

void test_check() {
	int nElem = 6;
	dim3 block(3);
	dim3 grid((nElem + block.x - 1) / block.x);

	printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
	printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

	checkIndex << <grid, block >> > ();

	cudaDeviceReset();

}

void test_change_block() {
	int nElem = 1024;
	dim3 block(1024);
	dim3 grid((nElem + block.x - 1) / block.x);
	printf("grid.x %d block.x %d\n", grid.x, block.x);


	block.x = 512;
	grid.x = (nElem + block.x - 1) / block.x;
	printf("grid.x %d block.x %d\n", grid.x, block.x);

	block.x = 256;
	grid.x = (nElem + block.x - 1) / block.x;
	printf("grid.x %d block.x %d\n", grid.x, block.x);

	block.x = 128;
	grid.x = (nElem + block.x - 1) / block.x;
	printf("grid.x %d block.x %d\n", grid.x, block.x);
}


