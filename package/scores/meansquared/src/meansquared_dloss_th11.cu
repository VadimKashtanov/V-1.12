#include "package/scores/meansquared/head/meansquared.cuh"

static __global__ void kernel_ms_dloss(
	float * grad, float * var, float * output,
	uint total, uint ostart, uint lines, uint outs,
	uint sets)
{
	uint out = threadIdx.x + blockIdx.x * blockDim.x;
	uint line = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	if (out < outs && line < lines)
	{
		uint pos = line*sets*total + set*total + ostart + out;
		grad[pos] = var[pos] - output[line*outs + out];
	};
};

void meansquared_dloss_th11(Train_t * train) {
	uint outpos = train->mdl->total - train->mdl->outputs;

	kernel_ms_dloss<<<dim3(KERN_DIV(train->mdl->outputs, 32), KERN_DIV(train->data->lines, 32), train->sets),dim3(32, 32, 1)>>>(
		train->_grad, train->_var, train->data->output_d,
		train->mdl->total, outpos, train->data->lines, train->data->outputs,
		train->sets
	);
	CUDA_WAIT_KER();
};