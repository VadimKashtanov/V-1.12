#include "package/scores/meansquared/head/meansquared.cuh"

static __global__ void opti_kernel_sum_scores_over_lines(
	float * grad, float * var, float * output,
	float * score_one_line_d,
	uint total, uint lines, uint sets, uint ostart, uint outs)
{
	uint out = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	if (out < outs)
	{
		float _sum_of_lines = 0;
		for (uint l=0; l < lines; l++) {
			_sum_of_lines += grad[l*sets*total + set*total + ostart + out];
		}
		score_one_line_d[set*outs + out] = _sum_of_lines;// / lines;
	};
};

static __global__ void opti_kernel_sum_scores_over_outputs(
	float * score_one_line_d, float * scores,
	uint total, uint sets, uint ostart, uint outs)
{
	uint set = blockIdx.x;

	uint start = set*outs + 0;
	float _sum_of_outs = 0;
	for (uint o=0; o < outs; o++) {
		_sum_of_outs += score_one_line_d[start];
		start++;
	}

	scores[set] = _sum_of_outs;// / outs;
};

void meansquared_score_th11(Train_t * train) {
	Mdl_t * mdl = train->mdl;

	uint outs = mdl->outputs;
	uint lines = train->data->lines;
	uint sets = train->sets;
	uint out_start = mdl->total - outs;

	//======================================================================
	//======================================================================

	//				sum over lines (only outputs)

	float * score_one_line_d;
	SAFE_CUDA(cudaMalloc((void**)&score_one_line_d, sizeof(float) * sets * outs));	//all lines are sumed in one (only outputs)

	opti_kernel_sum_scores_over_lines<<<dim3(KERN_DIV(outs, 16), sets),dim3(16,1)>>>(
		train->_grad, train->_var, train->data->output_d,
		score_one_line_d,
		mdl->total, lines, sets, out_start, outs);
	CUDA_WAIT_KER();

	//======================================================================
	//======================================================================

	//		sum of output pixels

	opti_kernel_sum_scores_over_outputs<<<dim3(sets),dim3(1)>>>(
		score_one_line_d, train->set_score_d,
		mdl->total, sets, out_start, outs);
	CUDA_WAIT_KER();

	SAFE_CUDA(cudaFree(score_one_line_d));

	//	Always copy set_score_d into Cpu Ram
	SAFE_CUDA(cudaMemcpy(train->set_score, train->set_score_d, sizeof(float) * sets, cudaMemcpyDeviceToHost));
};