#include "kernel/head/train.cuh"

void gtic_gtic(Train_t * train) {
	GTIC_GTIC[train->gtic->id](train);
};