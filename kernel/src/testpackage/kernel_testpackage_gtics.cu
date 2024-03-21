#include "kernel/head/testpackage.cuh"

void test_gtic(FILE * fp) {
	Train_t * train = test_package_load_train(fp);

	uint MIN_ECHOPES = read_uint(fp);
	is_123(fp);

	Mdl_t * mdl = train->mdl;
	Data_t * data = train->data;

	uint sets = train->sets;

	float * compare_array;

	for (uint echope=0; echope < MIN_ECHOPES; echope++) {
		gtic_gtic(train);
	}

	//dloss
	compare_array = load_float_array(mdl->weights*sets, fp);
	is_123(fp);

	train_print_compare_arr(train, "_weight", compare_array, 0.001);
	if (train_eq_arr(train, "_weight", compare_array, 0.001) == 0) {
		ERR("Pas d'egalité suffisante");
	}
	
	free(compare_array);
	
	train_free(train);
	data_free(data);
	mdl_free(mdl);
}