#include "kernel/head/train.cuh"

static void __global__ _add_x_1e5(
	uint w, uint ws, uint sets,
	float * _weight,
	float changement)
{
	uint set = threadIdx.x + blockIdx.x * blockDim.x;
	if (set < sets)
		_weight[set*ws + w] += changement;
};

static void __global__ _dSdw_1e5(
	uint w, uint ws,
	float * _meand,
	float * score_1e5, float * score,
	uint sets)
{
	uint set = threadIdx.x + blockIdx.x * blockDim.x;
	if (set < sets)
		_meand[set*ws + w] = 1e5 * (score_1e5[set] - score[set]);
};

//	dS = F(x+1e-5)-F(x)
void train_dSdw_1e5(Train_t * train, uint start_seed) {
	uint ws = train->mdl->weights;
	uint sets = train->sets;

	float * _score_plus_1e5_d;
	SAFE_CUDA(cudaMalloc((void**)&_score_plus_1e5_d, sizeof(float)*sets));
	for (uint w=0; w < ws; w++) {
		//	F(x+1e5)
		_add_x_1e5<<<dim3(KERN_DIV(sets, 32)),dim3(32)>>>(
			w, ws, sets,
			train->_weight,
			1e5);
		train_solo_compute_score(train, start_seed);
		SAFE_CUDA(cudaMemcpy(_score_plus_1e5_d, train->set_score_d, float(float)*sets, cudaMemcpyDeviceToDevice));
		_add_x_1e5<<<dim3(KERN_DIV(sets, 32)),dim3(32)>>>(
			w, ws, sets,
			train->_weight,
			-1e5);

		//	F(x+1e5)
		train_solo_compute_score(train, start_seed);

		//	f(x+)-f(x)
		_dSdw_1e5<<<dim3(KERN_DIV(train->sets, 32)),dim3(32)>>>(
			w, train->mdl->weights,
			train->_meand,
			_score_plus_1e5_d, train->set_score_d,
			train->sets
		);
		CUDA_WAIT_KER()
	}
	SAFE_CUDA(cudaFree(_score_plus_1e5_d));
};

//===============================================================================
//===============================================================================
//===============================================================================

static void __global__ _dSdwdw_1e5(
	uint dw, uint ws,
	float * _dd_weight,
	float * _meand_1e5, float * _meand,
	uint sets)
{
	uint w = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = threadIdx.y + blockIdx.y * blockDim.y;
	if (w < ws && set < sets)
		_dd_weight[dw*sets*ws + set*ws + w] = 1e5 * (_meand_1e5[set] - _meand[set]);
};

//	ddS = dF(x+1e-5)-dF(x)
void train_dSdwdw_1e5(Train_t * train, uint start_seed) {
	uint ws = train->mdl->weights;
	uint sets = train->sets;
	uint dws = train->dws;

	float * _meand_1e5;
	SAFE_CUDA(cudaMalloc((void**)&_meand_1e5, sizeof(float)*sets*ws));
	for (uint dw=0; dw < dws; dw++) {
		//	dF(x+1e-5)
		_add_x_1e5<<<dim3(KERN_DIV(sets, 32)),dim3(32)>>>(
			train->dw[dw], ws, sets,
			train->_weight,
			1e5);
		train_forward_backward(train, start_seed);
		SAFE_CUDA(cudaMemcpy(_meand_1e5, train->_meand, sizeof(float)*ws*sets, cudaMemcpyDeviceToDevice));
		_add_x_1e5<<<dim3(KERN_DIV(sets, 32)),dim3(32)>>>(
			train->dw[dw], ws, sets,
			train->_weight,
			-1e5);

		//	dF(x)
		train_forward_backward(train, start_seed);

		//	df(x+)-df(x)
		_dSdwdw_1e5<<<dim3(KERN_DIV(ws, 32), KERN_DIV(sets, 32)),dim3(32,32)>>>(
			train->dw[dw], ws,
			train->_dd_weight,
			_meand_1e5, train->_meand,
			train->sets
		);
		CUDA_WAIT_KER()
	}
	SAFE_CUDA(cudaFree(_meand_1e5));
};

//===============================================================================
//===============================================================================
//===============================================================================

static void __global__ _dSdwdw_1e10(
	uint w, uint dw, uint ws,
	float * _dd_weight,
	float * score_xy, float * score_x, float * score_y, float * score,
	uint sets)
{
	uint set = threadIdx.x + blockIdx.x * blockDim.x;
	if (set < sets)
		_dd_weight[dw*sets*ws + set*ws + w] = 1e10 * (score_xy[set] - score_x[set] - score_y[set] + score[set]);
};

//	ddS = F(x+1e-5,y+1e-5)-F(x+1e-5,y)-F(x,y+1e-5)+F(x,y)
void train_dSdwdw_1e10(Train_t * train, uint start_seed) {
	uint ws = train->mdl->weights;
	uint sets = train->sets;
	uint dws = train->dws;

	float * _score_plus_xy_d;
	SAFE_CUDA(cudaMalloc((void**)&_score_plus_xy_d, sizeof(float)*sets));

	float * _score_plus_x_d;
	SAFE_CUDA(cudaMalloc((void**)&_score_plus_x_d, sizeof(float)*sets));

	float * _score_plus_y_d;
	SAFE_CUDA(cudaMalloc((void**)&_score_plus_y_d, sizeof(float)*sets));

	for (uint dw=0; dw < dws; dw++) {
		for (uint w=0; w < ws; w++) {
			//	F(x+,y+)
			_add_x_1e5<<<dim3(KERN_DIV(sets, 32)),dim3(32)>>>(
				w, ws, sets, train->_weight, 1e5);
			_add_x_1e5<<<dim3(KERN_DIV(sets, 32)),dim3(32)>>>(
				train->dw[dw], ws, sets, train->_weight, 1e5);
			train_solo_compute_score(train, start_seed);
			SAFE_CUDA(cudaMemcpy(_score_plus_xy_d, train->set_score_d, float(float)*sets, cudaMemcpyDeviceToDevice));
			
			//	F(x+)
			_add_x_1e5<<<dim3(KERN_DIV(sets, 32)),dim3(32)>>>(
				train->dw[dw], ws, sets, train->_weight, -1e5);
			train_solo_compute_score(train, start_seed);
			SAFE_CUDA(cudaMemcpy(_score_plus_x_d, train->set_score_d, float(float)*sets, cudaMemcpyDeviceToDevice));
			
			//	F(y+)
			_add_x_1e5<<<dim3(KERN_DIV(sets, 32)),dim3(32)>>>(
				w, ws, sets, train->_weight, -1e5);
			_add_x_1e5<<<dim3(KERN_DIV(sets, 32)),dim3(32)>>>(
				train->dw[dw], ws, sets, train->_weight, 1e5);
			train_solo_compute_score(train, start_seed);
			SAFE_CUDA(cudaMemcpy(_score_plus_y_d, train->set_score_d, float(float)*sets, cudaMemcpyDeviceToDevice));

			//	F(x,y)
			_add_x_1e5<<<dim3(KERN_DIV(sets, 32)),dim3(32)>>>(
				train->dw[dw], ws, sets, train->_weight, -1e5);
			train_solo_compute_score(train, start_seed);
			train->set_score

			//	f(x+)-f(x)
			_dSdwdw_1e10<<<dim3(KERN_DIV(train->sets, 32)),dim3(32)>>>(
				w, dw, train->mdl->weights,
				train->_dd_weight,
				_score_plus_xy_d, _score_plus_x_d, _score_plus_y_d, train->set_score_d,
				train->sets
			);
			CUDA_WAIT_KER();
		}
	}
	SAFE_CUDA(cudaFree(_score_plus_xy_d));
	SAFE_CUDA(cudaFree(_score_plus_x_d));
	SAFE_CUDA(cudaFree(_score_plus_y_d));
};