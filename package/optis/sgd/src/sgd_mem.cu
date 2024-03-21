#include "package/optis/sgd/head/sgd.cuh"

void sgd_str_config(Config_t * config, char * key, char * value) {
	if (strcmp(key, "ALPHA") == 0) {
		config->param[0] = (float)atof(value);
	} else {
		ERR("What is %s (of value %s)", key, value);
	}
};

void sgd_mk(Train_t * train) {
	SGDData_t * ret = (SGDData_t*)malloc(sizeof(SGDData_t));
	ret->echopes = 0;
	memcpy(&ret->alpha, &train->opti->param[0], sizeof(float));
	train->opti->ptr = (void*)ret;
};

void sgd_free(Train_t * train) {
	free((SGDData_t*)train->opti->ptr);
	train->opti->ptr = 0;
};

void sgd_opti(Train_t * train) {
	sgd_opti_th11(train);
};