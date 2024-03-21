#include "kernel/head/testpackage.cuh"

void test_opti(FILE * fp) {
	Train_t * train = test_package_load_train(fp);

	uint MIN_ECHOPES = read_uint(fp);
	is_123(fp);

	Mdl_t * mdl = train->mdl;
	Data_t * data = train->data;

	uint sets = train->sets;

	float * compare_array;

	if (OPTI_require_ddf[train->opti->id]) assert(train->calcule_dd);

	for (uint echope=0; echope < MIN_ECHOPES; echope++) {
		if (train->calcule_dd) train_backward_of_forward_backward(train, 0);
		else train_forward_backward(train, 0);

		opti_opti(train);
	}

	//dloss
	compare_array = load_float_array(mdl->weights*sets, fp);
	is_123(fp);

	printf("=====================================================\n");
	printf("=========== Apres %i boucles d'optimisaiton =========\n", MIN_ECHOPES);
	printf("=====================================================\n");
	train_print_compare_arr(train, "_weight", compare_array, 0.001);
	if (train_eq_arr(train, "_weight", compare_array, 0.001) == 0) {
		ERR("Pas d'egalité suffisante");
	}
	
	free(compare_array);
	
	train_free(train);
	data_free(data);
	mdl_free(mdl);
}