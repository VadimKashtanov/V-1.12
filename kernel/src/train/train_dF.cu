#include "kernel/head/train.cuh"

void train_null_for_dS(Train_t * train) {
	//SAFE_CUDA(cudaMemset(train->_var, 0, sizeof(float) * train->sets * train->data->lines * train->mdl->vars));
	SAFE_CUDA(cudaMemset(train->_meand, 0, sizeof(float) * train->sets * train->mdl->weights))
	SAFE_CUDA(cudaMemset(train->_grad, 0, sizeof(float) * train->sets * train->data->lines * train->mdl->total))
};

void train_forward(Train_t * train, uint start_seed) {
	for (uint t=0; t < train->data->lines; t++) {
		for (uint i=0; i < train->mdl->insts; i++) {
			INST_FORWARD[train->mdl->inst[i]->id](train, train->mdl->inst[i], t, start_seed);
		}
	}
};

void train_backward(Train_t * train, uint start_seed) {
	for (int t=train->data->lines-1; t >= 0; t--) {
		for (int i=train->mdl->insts-1; i >= 0; i--) {
			INST_BACKWARD[train->mdl->inst[i]->id](train, train->mdl->inst[i], t, start_seed);
		}
	}
};
