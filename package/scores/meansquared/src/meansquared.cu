#include "package/scores/meansquared/head/meansquared.cuh"

void meansquared_str_config(Config_t * config, char * key, char * value) {
	ERR("There is no args for MEANSQUARED. (name=%s, value=%s)", key, value);
};

void meansquared_mk(Train_t * train) {
	
};

void meansquared_free(Train_t * train) {
	train->score->ptr = 0;
};

//	========================================

void meansquared_score(Train_t * train) {
	meansquared_score_th11(train);
};

void meansquared_loss(Train_t * train) {
	meansquared_loss_th11(train);
};

void meansquared_dloss(Train_t * train) {
	meansquared_dloss_th11(train);
};

void meansquared_ddloss(Train_t * train) {
	meansquared_ddloss_th11(train);
};