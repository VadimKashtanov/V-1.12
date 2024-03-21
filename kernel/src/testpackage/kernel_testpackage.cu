#include "kernel/head/testpackage.cuh"

float * load_float_array(uint len, FILE * fp) {
	float * ret = (float*)malloc(sizeof(float) * len);
	fread(ret, sizeof(float), len, fp);
	return ret;
};

//=====================================================================

static bool compare_arrays(float * cpu0, float * cpu1, uint count)
{
	for (uint i=0; i < count; i++) {
		if (compare_floats(cpu0[i], cpu1[i], 0.01) != true) {
			return false;
		}
	}
	return true;
};

bool test_package_compare_cpu_and_gpu(float * cpu0, float * gpu_d, uint count)
{
	float * cpu = (float*)malloc(sizeof(float) * count);
	SAFE_CUDA(cudaMemcpy(cpu, gpu_d, sizeof(float) * count, cudaMemcpyDeviceToHost));
	bool ret = compare_arrays(cpu0, cpu, count);
	free(cpu);
	return ret;
};

bool test_package_compare_cpu_and_cpu(float * cpu0, float * cpu1, uint count)
{
	return compare_arrays(cpu0, cpu1, count);
};

//==========================================================================

Data_t * load_test_data(FILE * fp)
{
	uint batchs, lines, inputs, outputs;

	fread(&batchs, sizeof(uint), 1, fp);
	fread(&lines, sizeof(uint), 1, fp);
	fread(&inputs, sizeof(uint), 1, fp);
	fread(&outputs, sizeof(uint), 1, fp);

	Data_t * ret = data_load(batchs, inputs, outputs, lines);

	data_cudamalloc(ret);

	fread(ret->input, sizeof(float), lines*inputs, fp);
	fread(ret->output, sizeof(float), lines*outputs, fp);

	data_cudamemcpy(ret);

	return ret;
};

Train_t * test_package_load_train(FILE * fp) {
	//	Load Contexte
	Mdl_t * mdl = mdl_fp_load(fp);
	is_123(fp);

	Data_t * data = load_test_data(fp);
	is_123(fp);

	uint dw[mdl->weights];
	for (uint i=0; i < mdl->weights; i++) dw[i] = i;

	uint calcule_d, calcule_dd;

	fread(&calcule_d, sizeof(uint), 1, fp);
	is_123(fp);

	fread(&calcule_dd, sizeof(uint), 1, fp);
	is_123(fp);

	Config_t * score = config_load(fp);
	Config_t * opti = config_load(fp);
	Config_t * gtic = config_load(fp);

	Train_t * train = mk_train(
		mdl, data,
		score, opti, gtic,//config_load(fp), config_load(fp), config_load(fp),
		calcule_d,
		calcule_dd,
		mdl->weights, //dws
		dw
	);
	train_random_weights(train, 0);

	float * compare_array = load_float_array(train->sets * mdl->weights, fp);
	is_123(fp);

	printf("==== On verifie simplement que train est initialisé avec les memes weights en python et c/cuda ====\n");
	train_print_compare_arr(train, "_weight", compare_array, 0.001);
	if (train_eq_arr(train, "_weight", compare_array, 0.001) == 0) {
		ERR("Pas egale");
	}

	free(compare_array);

	return train;
};