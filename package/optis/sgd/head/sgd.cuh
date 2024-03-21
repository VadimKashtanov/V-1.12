#pragma once

#include "kernel/head/train.cuh"

/*
Stocastic Gradient Descent.

Vannila or classic grandient descent. Only with a gradient step.

	w -= alpha * grad(w)
*/

typedef struct {
	uint echopes;
	float alpha;
} SGDData_t;

void sgd_str_config(Config_t * config, char * key, char * value);
void sgd_mk(Train_t * train);
void sgd_free(Train_t * train);

//	==== Gtic & Kernels ====
void sgd_opti_th11(Train_t * train);

void sgd_opti(Train_t * train);