#include "package/scores/meansquared/head/meansquared.cuh"

static __global__ void kernel_ms_ddloss(
	float * dd_grad, float * dd_var,
	uint total, uint ostart, uint lines, uint outs,
	uint sets)
{
	uint out = threadIdx.x + blockIdx.x * blockDim.x;
	uint line = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	if (out < outs && line < lines)
	{
		uint pos = line*sets*total + set*total + ostart + out;
		dd_var[pos] += dd_grad[pos];
	};
};

void meansquared_ddloss_th11(Train_t * train) {

	uint outpos = train->mdl->total - train->mdl->outputs;

	kernel_ms_ddloss<<<dim3(KERN_DIV(train->mdl->outputs, 32), KERN_DIV(train->data->lines, 32), train->sets),dim3(32, 32, 1)>>>(
		train->_dd_grad, train->_dd_var,
		train->mdl->total, outpos, train->data->lines, train->data->outputs,
		train->sets
	);
	
	CUDA_WAIT_KER();
};