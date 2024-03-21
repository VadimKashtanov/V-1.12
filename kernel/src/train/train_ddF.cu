#include "kernel/head/train.cuh"

void train_ddS_first_null(Train_t * train) {
	Mdl_t * mdl = train->mdl;

	//SAFE_CUDA(cudaMemset(train->_var, 0, sizeof(float) * train->sets * train->data->lines * mdl->total));
	SAFE_CUDA(cudaMemset(train->_meand, 0, sizeof(float) * train->sets * mdl->weights));
	SAFE_CUDA(cudaMemset(train->_grad, 0, sizeof(float) * train->sets * train->data->lines * mdl->total));
	//SAFE_CUDA(cudaMemset(train->_locd, 0, sizeof(float) * train->sets * train->data->lines * mdl->locds));
	//SAFE_CUDA(cudaMemset(train->_locd2, 0, sizeof(float) * train->sets * train->data->lines * mdl->locds2));
};

void train_forward2(Train_t * train, uint start_seed) {
	for (uint t=0; t < train->data->lines; t++) {
		for (uint i=0; i < train->mdl->insts; i++) {
			INST_FORWARD2[train->mdl->inst[i]->id](train, train->mdl->inst[i], t, start_seed);
		}
	}
};

void train_backward2(Train_t * train, uint start_seed) {
	for (int t=train->data->lines-1; t >= 0; t--) {
		for (int i=train->mdl->insts-1; i >= 0; i--) {
			INST_BACKWARD2[train->mdl->inst[i]->id](train, train->mdl->inst[i], t, start_seed);
		}
	}
};

void train_ddS_second_dd_restart(Train_t * train, uint dw) {
	Mdl_t * mdl = train->mdl;
	
	SAFE_CUDA(cudaMemset(train->_dd_weight + dw*train->sets * mdl->weights, 0, sizeof(float) * train->sets * mdl->weights));	//tous vont etre memset
	SAFE_CUDA(cudaMemset(train->_dd_var, 0, sizeof(float) * train->sets * train->data->lines * mdl->total));
	SAFE_CUDA(cudaMemset(train->_dd_meand, 0, sizeof(float) * train->sets * mdl->weights));
	SAFE_CUDA(cudaMemset(train->_dd_grad, 0, sizeof(float) * train->sets * train->data->lines * mdl->total));
	SAFE_CUDA(cudaMemset(train->_dd_locd, 0, sizeof(float) * train->sets * train->data->lines * mdl->locds));

	float _1 = 1.0;
	//_dd_meand[i] = 1;
	for (uint set=0; set < train->sets; set++)
		SAFE_CUDA(cudaMemcpy(train->_dd_meand + set*train->mdl->weights + train->dw[dw], &_1, sizeof(float) * 1, cudaMemcpyHostToDevice));
};

void train_backward_of_backward2(Train_t * train, uint dw, uint start_seed) {
	for (uint ligne=0; ligne < train->data->lines; ligne++) {
		for (uint inst=0; inst < train->mdl->insts; inst++) {
			INST_BACKWARD_OF_BACKWARD2[train->mdl->inst[inst]->id](train, train->mdl->inst[inst], dw, ligne, start_seed);
		}
	}
};

void train_backward_of_forward2(Train_t * train, uint dw, uint start_seed) {
	for (int ligne=train->data->lines-1; ligne >= 0; ligne--) {
		for (int inst=train->mdl->insts-1; inst >= 0; inst--) {
			INST_BACKWARD_OF_FORWARD2[train->mdl->inst[inst]->id](train, train->mdl->inst[inst], dw, ligne, start_seed);
		}
	}
};