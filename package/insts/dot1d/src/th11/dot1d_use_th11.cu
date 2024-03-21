#include "package/insts/dot1d/head/dot1d.cuh"

__global__
void dot1d_use_th11(
	uint Ax, uint Yx,
	uint activ,
	uint time,
	uint total,
	uint istart, uint ystart, uint wstart,
	float * var, float * weight)
{
	uint _Yx = threadIdx.x + blockIdx.x*blockDim.x;

	if (_Yx < Yx) {
		uint Apos = time*total + istart;
		uint Bpos = wstart + _Yx;	//	Dot1d does not store W as Dot2d       in fact Dot2D.T = Dot1d  (it would be better to change it)

		float sum = 0;
		for (uint i=0; i < Ax; i++) {
			sum += var[Apos] * weight[Bpos];
			//printf("%i, %i, %f, %f", _Yx, i, var[Apos], weight[Bpos]);
			Apos++;
			Bpos += Yx;
		}
		sum += weight[wstart + Yx*Ax + _Yx];

		ACTIVATION_USE(activ, sum, sum);

		var[time*total + ystart + _Yx] = sum;

		//printf("%f\n", var[time*total + ystart + _Yx]);
	}
};