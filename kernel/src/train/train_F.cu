#include "kernel/head/train.cuh"

static __global__ void kernel_set_input(float * _var, float * _input, uint total, uint sets, uint inputs, uint lines) {
	uint _inp = threadIdx.x + blockIdx.x * blockDim.x,	\
		 line = threadIdx.y + blockIdx.y * blockDim.y,	\
		 set = blockIdx.z;

	if (_inp < inputs && line < lines) {
		_var[line*sets*total + set*total + _inp] = _input[line*inputs + _inp];
	}
};

void train_set_input(Train_t * train) {
	kernel_set_input<<<dim3(KERN_DIV(train->mdl->inputs,32), KERN_DIV(train->data->lines,32), train->sets),dim3(32,32,1)>>>(
		train->_var, train->data->input_d, train->mdl->total, train->sets, train->mdl->inputs, train->data->lines);
	SAFE_CUDA(cudaPeekAtLastError());
};

void train_restart(Train_t * train) {
	SAFE_CUDA(cudaMemset(train->_var, 0, sizeof(float) * train->sets * train->mdl->weights));
};