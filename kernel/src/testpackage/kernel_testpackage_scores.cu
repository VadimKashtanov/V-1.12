#include "kernel/head/testpackage.cuh"

void test_score(FILE * fp) {
	Train_t * train = test_package_load_train(fp);

	Mdl_t * mdl = train->mdl;
	Data_t * data = train->data;

	uint lines = data->lines;

	uint sets = train->sets;

	float * compare_array;

	train_set_input(train);
	train_null_for_dS(train);
	train_forward(train, 0);

	//dloss
	score_dloss(train);
	compare_array = load_float_array(mdl->total*lines*sets, fp);
	is_123(fp);

	train_print_compare_arr(train, "_grad", compare_array, 0.001);
	if (train_eq_arr(train, "_grad", compare_array, 0.001) == 0) {
		ERR("Pas d'egalité suffisante");
	}
	
	free(compare_array);

	//loss
	score_loss(train);
	compare_array = load_float_array(mdl->total*lines*sets, fp);
	is_123(fp);

	train_print_compare_arr(train, "_grad", compare_array, 0.001);
	if (train_eq_arr(train, "_grad", compare_array, 0.001) == 0) {
		ERR("Pas d'egalité suffisante");
	}

	free(compare_array);

	//score
	score_score(train);
	compare_array = load_float_array(sets, fp);
	is_123(fp);

	score_compare_scores(train, compare_array, 0.001);
	if (score_eq_score(train, compare_array, 0.001) == 0) {
		ERR("Pas d'egalité suffisante");
	}

	free(compare_array);


	//======================== ddf

	if (SCORE_allow_ddf[train->score->id]) assert(train->calcule_dd);

	if (train->calcule_dd && SCORE_allow_ddf[train->score->id]) {
		train_ddS_first_null(train);

		train_forward2(train, 0);
		score_dloss(train);
		train_backward2(train, 0);

		//	C'est comme si on derivait plusieurs loss functions ou la loss function c'est S = (meand[w] - 0), donc d(dS/meand[w])/dmeand[w] = 1
		uint dw = 0;
		train_ddS_second_dd_restart(train, dw);

		train_backward_of_backward2(train, dw, 0);
		score_ddloss(train);

		compare_array = load_float_array(mdl->total*lines*sets, fp);
		is_123(fp);

		train_print_compare_arr(train, "_dd_grad", compare_array, 0.001);
		if (train_eq_arr(train, "_dd_grad", compare_array, 0.001) == 0) {
			ERR("Pas d'egalité suffisante");
		}
		
		free(compare_array);
	}
	
	train_free(train);
	data_free(data);
	mdl_free(mdl);
}