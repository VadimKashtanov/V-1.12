#include "package/scores/meansquared/head/meansquared.cuh"

static __global__ void opti_kernel_ms_loss(
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
		float g = var[pos];
		float w = output[line*outs + out];
		grad[pos] = pow(g - w, 2)/2;
	};
};

void meansquared_loss_th11(Train_t * train) {
	Mdl_t * mdl = train->mdl;

	uint outs = mdl->outputs;
	uint lines = train->data->lines;
	uint sets = train->sets;
	uint out_start = mdl->total - outs;

	//======================================================================

	//						compute score

	opti_kernel_ms_loss<<<dim3(KERN_DIV(outs, 32), KERN_DIV(lines, 32), sets),dim3(32,32,1)>>>(
		train->_grad, train->_var, train->data->output_d,
		mdl->total, out_start, lines, outs,
		sets);
	CUDA_WAIT_KER();
};