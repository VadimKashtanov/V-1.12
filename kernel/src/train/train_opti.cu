#include "kernel/head/train.cuh"

void opti_opti(Train_t * train) {
	OPTI_OPTI[train->opti->id](train);
};