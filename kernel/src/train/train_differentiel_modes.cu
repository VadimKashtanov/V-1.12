#include "kernel/head/train.cuh"

void train_forward_backward(Train_t * train, uint start_seed) {
	if (train->mdl->capable_df != 1) ERR("train->mdl->capable_df == %i", train->calcule_d);

	train_null_for_dS(train);
	train_restart(train);
	train_set_input(train);

	train_forward(train, start_seed);
	score_dloss(train);
	train_backward(train, start_seed);
};

void train_backward_of_forward_backward(Train_t * train, uint start_seed) {
	if (train->mdl->capable_ddf != 1) ERR("train->mdl->capable_ddf == %i", train->calcule_dd);

	train_restart(train);
	train_ddS_first_null(train);
	train_set_input(train);

	train_forward2(train, start_seed);
	score_dloss(train);
	train_backward2(train, start_seed);

	//	C'est comme si on derivait plusieurs loss functions ou la loss function c'est S = (meand[w] - 0), donc d(dS/meand[w])/dmeand[w] = 1
	for (uint dw=0; dw < train->dws; dw++) {
		train_ddS_second_dd_restart(train, dw);

		train_backward_of_backward2(train, dw, start_seed);
		score_ddloss(train);
		train_backward_of_forward2(train, dw, start_seed);
		//train_deriv_set_input(train);	// pas utile pour dwidwj
	}
};