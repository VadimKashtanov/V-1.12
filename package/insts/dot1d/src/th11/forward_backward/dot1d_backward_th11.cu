#include "package/insts/dot1d/head/dot1d.cuh"

__global__
void dot1d_backward_th11(
	uint Ax, uint Yx,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint sets)
{
	/*	Kernel coordinates	*/
	uint _Yx = threadIdx.x + blockIdx.x*blockDim.x, \
		 set = blockIdx.y;

	if (_Yx < Yx) {
		uint Apos = time*sets*total + set*total + istart;
		//uint weight_start = set*wsize + wstart;
		uint Bpos = set*wsize + wstart + _Yx;// _Yx*Ax;

		float dlds = locd[time*sets*lsize + set*lsize + lstart + _Yx] * grad[time*sets*total + set*total + ystart + _Yx];

		meand[set*wsize + wstart + Yx*Ax + _Yx] += dlds;

		for (uint i=0; i < Ax; i++) {
			//if (pseudo_randomf(Apos*(seed+1)) >= drop_rate) {
			atomicAdd(&grad[Apos], dlds * weight[Bpos]);
			atomicAdd(&meand[Bpos], dlds * var[Apos]);
			//}
			Apos++;
			Bpos += Yx;
		}
	}
};