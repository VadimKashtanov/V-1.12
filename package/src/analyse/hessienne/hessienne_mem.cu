#include "kernel/head/analyse/hessienne.cuh"

Hessienne_t * mk_hessienne(Opti_t * opti) {
	Hessienne_t * ret = (Hessienne_t*)malloc(sizeof(Hessienne_t));

	ret->opti = opti;

	uint sets = opti->train->sets;
	uint wsize = opti->train->mdl->weights;

	//ret->tableau_d = 0;
	ret->tableau = (float*)malloc(sizeof(float) * sets * wsize * wsize);

	ret->inverse_par_set_d = 0;
	ret->inverse_par_set = (float*)malloc(sizeof(float) * sets * wsize * wsize);

	return ret;
};

void cudmalloc_hessienne(Hessienne_t * hessienne) {
	uint sets = hessienne->opti->train->sets;
	uint wsize = hessienne->opti->train->mdl->weights;

	//SAFE_CUDA(cudaMalloc((void**)&hessienne->tableau_d, sizeof(float) * sets * wszie * wszie))
	SAFE_CUDA(cudaMalloc((void**)&hessienne->inverse_par_set_d, sizeof(float) * sets * wsize * wsize))
};

void free_hessienne(Hessienne_t * hessienne) {
	//if (hessienne->tableau_d) SAFE_CUDA(cudaFree(hessienne->tableau_d));
	if (hessienne->inverse_par_set_d) SAFE_CUDA(cudaFree(hessienne->inverse_par_set_d));

	free(hessienne->tableau);
	free(hessienne->inverse_par_set);

	free(hessienne);
};