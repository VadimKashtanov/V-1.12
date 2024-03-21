#include "kernel/head/train.cuh"

//	---------------------------------------------------------------------------------------------
//	------------------------------------ Aleatoire ----------------------------------------------
//	---------------------------------------------------------------------------------------------	

static __global__ void kernel_random_weights(uint rnd_seed, uint weights, float * _weight) {
	uint wid = threadIdx.x + blockIdx.x*blockDim.x;
	uint set = threadIdx.y + blockIdx.y*blockDim.y;
	uint pos = set*weights + wid;

	if (wid < weights) {
		_weight[pos] = pseudo_randomf_minus1_1(rnd_seed + pos);
	}
};

void train_random_weights(Train_t * train, uint rnd_seed) {
	kernel_random_weights<<<dim3(KERN_DIV(train->mdl->weights,32), train->sets), dim3(32,1)>>>(
		rnd_seed, train->mdl->weights, train->_weight);
	CUDA_WAIT_KER()
};

//	---------------------------------------------------------------------------------------------
//	----------------------------- Random Weight From Mdl ----------------------------------------
//	---------------------------------------------------------------------------------------------	

static __global__ void kernel_random_weights_from_mdl(uint rnd_seed, uint weights, float * _weight, float * mdl_weight_d, float coef) {
	uint wid = threadIdx.x + blockIdx.x*blockDim.x;
	uint set = threadIdx.y + blockIdx.y*blockDim.y;
	uint pos = set*weights + wid;

	if (wid < weights) {
		uint a = rnd_seed + pos;
		_weight[pos] = mdl_weight_d[wid] + coef*pseudo_randomf_minus1_1(a);
	}
};

void train_random_weights_from_mdl(Train_t * train, uint rnd_seed, float coef) {
	float * mdl_weights_d;
	SAFE_CUDA(cudaMalloc((void**)&mdl_weights_d, sizeof(float)*train->mdl->weights));
	SAFE_CUDA(cudaMemcpy(mdl_weights_d, train->mdl->weight, sizeof(float)*train->mdl->weights, cudaMemcpyHostToDevice));

	kernel_random_weights_from_mdl<<<dim3(KERN_DIV(train->mdl->weights, 32), train->sets),dim3(32,1)>>>(
		rnd_seed, train->mdl->weights, train->_weight, mdl_weights_d, coef);	//coef = 0.1 c'est pas mal
	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());

	SAFE_CUDA(cudaFree(mdl_weights_d));
};

//	---------------------------------------------------------------------------------------------
//	----------------------- Injecter Parametres vers un set -------------------------------------
//	---------------------------------------------------------------------------------------------

void train_inject_weight_from_mdl_to_one_set(Train_t * train, uint set) {
	SAFE_CUDA(cudaMemcpy(
		train->_weight + set*train->mdl->weights, train->mdl->weight,
		sizeof(float)*train->mdl->weights, cudaMemcpyHostToDevice));
};

void train_inject_weight_cpu_to_one_set(Train_t * train, float * weight, uint set) {
	SAFE_CUDA(cudaMemcpy(
		train->_weight + set*train->mdl->weights, weight,
		sizeof(float)*train->mdl->weights, cudaMemcpyHostToDevice));
};

void train_inject_weight_gpu_to_one_set(Train_t * train, float * weight_d, uint set) {
	SAFE_CUDA(cudaMemcpy(
		train->_weight + set*train->mdl->weights, weight_d,
		sizeof(float)*train->mdl->weights, cudaMemcpyDeviceToDevice));
};

//	---------------------------------------------------------------------------------------------
//	------------------------------ Train->weight -> Train->mdl ----------------------------------
//	---------------------------------------------------------------------------------------------

void train_cpy_ws_to_mdl(Train_t * train, uint set) {
	SAFE_CUDA(cudaMemcpy(
		train->mdl->weight, train->_weight + set*train->mdl->weights,
		sizeof(float)*train->mdl->weights, cudaMemcpyDeviceToHost));
};