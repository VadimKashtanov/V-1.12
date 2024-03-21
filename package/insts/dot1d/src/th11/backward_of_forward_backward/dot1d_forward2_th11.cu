#include "package/insts/dot1d/head/dot1d.cuh"

__global__
void dot1d_forward2_th11(
	uint Ax, uint Yx, uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart, uint l2start,
	uint total, uint wsize, uint lsize, uint l2size,
	float * var, float * weight, float * locd, float * locd2,
	uint sets)
{
	//	Toutes les arrays _dd_ sont relatives a dS/dw[i] (dw)
	
	uint x = threadIdx.x + blockIdx.x*blockDim.x, \
		 set = blockIdx.y;

	if (x < Yx) {

		uint Apos;// = time*sets*total + set*total + input_start;
		uint Bpos;// + _Yx*Ax;

		//uint __seed;
		//float value;

		float sum = 0;
		for (uint i=0; i < Ax; i++) {

			Apos = time*sets*total + set*total + istart + i;
			Bpos = set*wsize + wstart + x + i*Yx;

			//__seed = Apos * (seed+1);

			//value = pseudo_randomf(__seed);

			//if ((value) >= drop_rate) {
			sum += var[Apos] * weight[Bpos];
			//}
		}
		sum += weight[set*wsize + wstart + Ax*Yx + x];
		
		float __locd, __locd2;

		ACTIVATION_FORWARD_2(activ, sum, sum, __locd, __locd2);

		var[time*sets*total + set*total + ystart + x] = sum;		//same assembler than putting it in if/else structure
		locd[time*sets*lsize + set*lsize + lstart + x] = __locd;
		locd2[time*sets*l2size + set*l2size + l2start + x] = __locd2;
	}
}