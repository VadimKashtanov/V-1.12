#include "kernel/head/analyse/hessienne.cuh"

//x -= H**(-1) * grad(x)

/*
	|x0|	| h0 h1 h2 |   | dx0 |
	|x1| -= | h3 h4 h5 | @ | dx1 |
	|x2|	| h6 h7 h8 |   | dx2 |
*/

static __global__ void appliquer_modification(float * weight, float * H_1, float * grad, uint sets, uint wsize, float alpha) {
	uint w = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	if (w < wsize) {
		float somme = 0;

		for (uint i=0; i < wsize; i++)
			somme += H_1[set*wsize*wsize + w*wsize + i] * grad[set*wsize + i];

		weight[set*wsize + w] -= alpha * somme;
	};
};

void opti_hessienne(Hessienne_t * hessienne) {
	Train_t * train = hessienne->opti->train;
	Mdl_t * mdl = train->mdl;

	uint wsize = mdl->weights;

	float alpha = 1.0;

	//	Obtention du gradient
	train_set_input(train);
	train_forward(train, 0);
	opti_dloss(hessienne->opti);
	train_backward(train, 0);

	//	
	appliquer_modification<<<dim3(KERN_DIV(wsize, 32),train->sets),dim3(32, 1)>>>(
		train->_weight, hessienne->inverse_par_set_d, train->_meand, train->sets, wsize, alpha
	);

	CUDA_WAIT_KER()
};