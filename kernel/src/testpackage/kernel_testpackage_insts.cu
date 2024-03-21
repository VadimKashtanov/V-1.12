#include "kernel/head/testpackage.cuh"

void test_inst(FILE * fp) {
	float * compare_array;
	
	Train_t * train = test_package_load_train(fp);

	Mdl_t * mdl = train->mdl;
	Data_t * data = train->data;

	uint total = mdl->total;
	uint lines = data->lines;
	uint locds = mdl->locds;
	uint locd2s = mdl->locd2s;
	uint weights = mdl->weights;

	//	Cpu et Use
	compare_array = load_float_array(total * lines, fp);
	is_123(fp);

	printf("==========================================\n");
	printf("================= Cpu_t ==================\n");
	printf("==========================================\n");

	Cpu_t * cpu = cpu_mk(mdl, data);

	cpu_set_input(cpu);
	cpu_forward(cpu);
		
	//	Compare
	cpu_compare_vars(cpu, compare_array);
	if (test_package_compare_cpu_and_cpu(compare_array, cpu->var, mdl->total*data->lines)) {
		OK("Cpu_t->var passed correctly.")
	} else {
		ERR("Il y a des Erreures avec Cpu_t->var")
	}

	cpu_free(cpu);

	printf("==========================================\n");
	printf("================= Use_t ==================\n");
	printf("==========================================\n");

	Use_t * use = use_mk(mdl, data);

	use_set_input(use);
	use_forward(use);

	//	Compare
	use_compare_vars(use, compare_array);
	if (test_package_compare_cpu_and_gpu(compare_array, use->var_d, mdl->total*data->lines)) {
		OK("Use_t->var_d passed correctly.")
	} else {
		ERR("Il y a des Erreures avec Use_t->var_d")
	}

	use_free(use);

	free(compare_array);

	printf("==========================================\n");
	printf("================= Train_t ==================\n");
	printf("==========================================\n");

	uint sets = train->sets;

	const char * arr_names[11] = {
		"_weight",
		"_var",
		
		"_grad",
		"_locd",
		"_meand",
		
		"_dd_weight",
		"_dd_var",
		"_dd_meand",
		"_dd_grad",
		"_dd_locd",
		"_locd2"
	};

	uint lens[11] = {
		sets * weights,
		lines * sets * total,
		lines * sets * total,
		lines * sets * locds,
		sets * weights,

		train->dws * sets * weights,
		lines * sets * total,
		sets * weights,
		lines * sets * total,
		lines * sets * locds,
		lines * sets * locd2s,
	};

	if (train->calcule_d == 1) {
		train_forward_backward(train, 0);

		for (uint i=0; i < 5; i++) {
			printf("=========================================\n");
			printf("=============== %s ============\n", arr_names[i]);

			compare_array = load_float_array(lens[i], fp);
			is_123(fp);

			train_print_compare_arr(train, arr_names[i], compare_array, 0.001);
			if (train_eq_arr(train, arr_names[i], compare_array, 0.001) == 0) {
				ERR("Pas d'egalité suffisante");
			}
		}

		//	On teste juste 1e5 les dSdw. Histoire de reverifier encore un fois
		float * _meand = (float*)malloc(sizeof(float) * weights * sets);
		SAFE_CUDA(cudaMemcpy(_meand, train->_meand, sizeof(float)*weights*sets, cudaMemcpyDeviceToHost));

		printf("=========================================\n");
		printf("============ Tester _meand 1e5 ==========\n");
		train_dSdw_1e5(train, 0);
		//
		train_print_compare_arr(train, "_meand", _meand, 0.001);
		printf("     dw 1e5 ----- dS (forward-backward)\n");
		if (train_eq_arr(train, "_meand", compare_array, 0.001) == 0) {
			ERR("Pas d'egalité suffisante");
		}

		free(_meand);
	}

	if (train->calcule_dd == 1) {
		train_backward_of_forward_backward(train, 0);

		for (uint i=0; i < 11; i++) {
			printf("=========================================\n");
			printf("=============== %s ============\n", arr_names[i]);

			compare_array = load_float_array(lens[i], fp);
			is_123(fp);

			train_print_compare_arr(train, arr_names[i], compare_array, 0.001);
			if (train_eq_arr(train, arr_names[i], compare_array, 0.001) == 0) {
				ERR("Pas d'egalité suffisante");
			}

			free(compare_array);
		}


		//	On teste juste 1e10 les d(dSdwi)/dwj. Histoire de reverifier encore un fois
		float * _dd_weight = (float*)malloc(sizeof(float) * weights * sets * train->dws);
		SAFE_CUDA(cudaMemcpy(_dd_weight, train->_dd_weight, sizeof(float)*train->dws*weights*sets, cudaMemcpyDeviceToHost));

		printf("===========================================\n");
		printf("========= Tester _dd_weight 1e10 ==========\n");
		train_dSdwdw_1e10(train, 0);
		//
		train_print_compare_arr(train, "_dd_weight", _dd_weight, 0.001);
		printf("      dwdw 1e10 ----- ddS (forward-backward ** 2)\n");
		if (train_eq_arr(train, "_dd_weight", _dd_weight, 0.001) == 0) {
			ERR("Pas d'egalité suffisante");
		}
		
		free(_dd_weight);
	}

	train_free(train);
	data_free(data);
	mdl_free(mdl);
}