#include "package/insts/dot1d/head/dot1d.cuh"

//y = ax + b
//Backward:
//da += dy * x
//dx += dy * a
//db += dy

__global__
void dot1d_backward_of_backward2_th11(
	uint Ax, uint Yx, uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	float * _dd_var, float * _dd_weight, float * _dd_locd, float * _dd_grad, float * _dd_meand,
	uint sets)
{
	//	Toutes les arrays _dd_ sont relatives a dS/dw[i] (dw)
	
	/*	Kernel coordinates	*/
	uint _Yx = threadIdx.x + blockIdx.x*blockDim.x, \
		 set = blockIdx.y;

	if (_Yx < Yx) {
		uint Apos = time*sets*total + set*total + istart;
		//uint weight_start = set*wsize + wstart;
		uint Bpos = set*wsize + wstart + _Yx;// _Yx*Ax;

		uint lpos = time*sets*lsize + set*lsize + lstart + _Yx;
		uint ypos = time*sets*total + set*total + ystart + _Yx;

		float dlds = locd[lpos] * grad[ypos];

		//meand[wstart + Yx*Ax + _Yx] += dlds;
		float D_dlds = _dd_meand[set*wsize + wstart + Yx*Ax + _Yx];

		float tmp;
		for (uint i=0; i < Ax; i++) {
			//atomicAdd(&grad[Apos], dlds * weight[Bpos]);
			tmp = _dd_grad[Apos];
			if (tmp != 0) {
				D_dlds += tmp * weight[Bpos];
				atomicAdd(&_dd_weight[Bpos], tmp * dlds);
			}

			//atomicAdd(&meand[Bpos], dlds * var[Apos]);
			tmp = _dd_meand[Bpos];
			if (tmp != 0) {
				D_dlds += tmp * var[Apos];
				atomicAdd(&_dd_var[Apos], tmp * dlds);
			}
			
			Apos++;
			Bpos += Yx;
		}

		//Backward of
		//float dlds = locd[time*sets*lsize + set*lsize + lstart + _Yx] * grad[time*sets*total + set*total + ystart + _Yx];
		if (D_dlds != 0) {
			atomicAdd(&_dd_grad[ypos], D_dlds * locd[lpos]);
			atomicAdd(&_dd_locd[lpos], D_dlds * grad[ypos]);
		}
	}
};