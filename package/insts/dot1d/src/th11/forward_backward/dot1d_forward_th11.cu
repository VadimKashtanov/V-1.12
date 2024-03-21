#include "package/insts/dot1d/head/dot1d.cuh"

__global__
void dot1d_forward_th11(
	uint Ax, uint Yx,
	uint activ,
	uint time,
	uint input_start, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd,
	uint sets)
{
	uint x = threadIdx.x + blockIdx.x*blockDim.x, \
		 set = blockIdx.y;

	if (x < Yx) {

		uint Apos;// = time*sets*total + set*total + input_start;
		uint Bpos;// + _Yx*Ax;

		//uint __seed;
		//float value;

		float sum = 0;
		for (uint i=0; i < Ax; i++) {

			Apos = time*sets*total + set*total + input_start + i;
			Bpos = set*wsize + wstart + x + i*Yx;

			//__seed = Apos * (seed+1);

			//value = pseudo_randomf(__seed);

			//if ((value) >= drop_rate) {
			sum += var[Apos] * weight[Bpos];
			//}
		}
		sum += weight[set*wsize + wstart + Ax*Yx + x];
		
		float __locd;

		ACTIVATION_FORWARD(activ, sum, sum, __locd)

		var[time*sets*total + set*total + ystart + x] = sum;		//same assembler than putting it in if/else structure
		locd[time*sets*lsize + set*lsize + lstart + x] = __locd;
	}
};
