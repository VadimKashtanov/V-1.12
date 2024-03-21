#include "package/insts/dot1d/head/dot1d.cuh"

__global__
void dot1d_backward_of_forward2_th11(
	uint Ax, uint Yx,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart, uint l2start,
	uint total, uint wsize, uint lsize, uint l2size,
	float * var, float * weight, float * locd, float * locd2, float * grad, float * meand,
	float * _dd_var, float * _dd_weight, float * _dd_locd, float * _dd_grad, float * _dd_meand,
	uint sets)
{
	//	Toutes les arrays _dd_ sont relatives a dS/dw[i] (dw)

	uint x = threadIdx.x + blockIdx.x*blockDim.x, \
		 set = blockIdx.y;

	if (x < Yx) {

		uint Apos;// = time*sets*total + set*total + input_start;
		uint Bpos;// + _Yx*Ax;

		/*float sum = 0;
		for (uint i=0; i < Ax; i++) {

			Apos = time*sets*total + set*total + input_start + i;
			Bpos = set*wsize + wstart + x + i*Yx;

			//sum += var[Apos] * weight[Bpos];
		}
		sum += weight[set*wsize + wstart + Ax*Yx + x];
		
		float __locd, __locd2;

		var[time*sets*total + set*total + ystart + x] = sum;		//same assembler than putting it in if/else structure
		locd[time*sets*lsize + set*lsize + lstart + x] = __locd;
		locd2[time*sets*l2size + set*l2size + l2start + x] = __locd2;*/

		//__locd(s) est fonction de s, donc il faut derivee s. Voir le doc fondateur.
		//__locd2 est pas lié a forward_backward, il vient en appuis. On derive que le `var = sum` et `locd = __locd(s)`
		//car seul locd joue dans une des multiplication 

		uint ypos = time*sets*total + set*total + ystart + x;
		uint lpos = time*sets*lsize + set*lsize + lstart + x;
		uint l2pos = time*sets*l2size + set*l2size + l2start + x;

		float ds = 0;
		ds += locd2[l2pos] * _dd_locd[lpos];
		ds += locd[lpos] * _dd_var[ypos];

		atomicAdd(&_dd_weight[set*wsize + wstart + Ax*Yx + x], ds);

		if (ds != 0) {
			for (uint k=0; k < Ax; k++) {
				Apos = time*sets*total + set*total + istart + k;
				Bpos = set*wsize + wstart + x + k*Yx;

				//sum += var[Apos] * weight[Bpos];
				atomicAdd(&_dd_var[Apos], ds * weight[Bpos]);
				atomicAdd(&_dd_weight[Bpos], ds * var[Apos]);
			}
		}
	}
}