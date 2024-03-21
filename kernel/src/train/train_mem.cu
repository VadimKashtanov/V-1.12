#include "kernel/head/train.cuh"

Train_t* mk_train(
	Mdl_t * mdl, Data_t * data,
	const Config_t * score, const Config_t * opti, const Config_t * gtic,
	const uint calcule_d,
	const uint calcule_dd,
	const uint dws, const uint * dw)
{
	Train_t * ret = (Train_t*)malloc(sizeof(Train_t));

	//	Mdl & Data
	ret->mdl = mdl;
	ret->data = data;
	ret->lines = data->lines;

	if (mdl->inputs != data->inputs) ERR("Mdl->inputs (%i) != Data->inputs (%i)", mdl->inputs, data->inputs);
	if (mdl->outputs != data->outputs) ERR("Mdl->outputs (%i) != Data->outputs (%i)", mdl->outputs, data->outputs);

	if (!mdl->capable_df && calcule_d) ERR("mdl->capable_df (%i) != calcule_dF (%i)", mdl->capable_df, calcule_d);
	if (!mdl->capable_ddf && calcule_dd) ERR("mdl->capable_ddf (%i) != calcule_ddF (%i)", mdl->capable_ddf, calcule_dd);

	//	Sets & calcule_dd
	ret->sets = 0;
	ret->calcule_d = calcule_d;
	ret->calcule_dd = calcule_dd;
	ret->dws = dws;
	ret->dw = (uint*)malloc(sizeof(uint) * dws);
	memcpy(ret->dw, dw, sizeof(uint) * dws);
	
	//	Score
	ret->score = cpy_config(score);
	SCORE_MK[score->id](ret);

	//	Opti
	ret->opti = cpy_config(opti);
	OPTI_MK[opti->id](ret);

	//	Gtic
	ret->gtic = cpy_config(gtic);
	GTIC_MK[gtic->id](ret);

	if (ret->sets == 0) ERR("Le Gtic devait donner un nombre de sets non null");
	uint sets = ret->sets;

	uint lines = data->lines;
	uint ws = mdl->weights;

	SAFE_CUDA(cudaMalloc((void**)&ret->_weight, sizeof(float) * (ws*sets)));
	SAFE_CUDA(cudaMalloc((void**)&ret->_var, sizeof(float) * (mdl->total*sets*lines)));

	if (calcule_d) SAFE_CUDA(cudaMalloc((void**)&ret->_locd, sizeof(float) * (mdl->locds*sets*lines)));
	SAFE_CUDA(cudaMalloc((void**)&ret->_grad, sizeof(float) * (mdl->total*sets*lines)));
	SAFE_CUDA(cudaMalloc((void**)&ret->_meand, sizeof(float) * (ws*sets)));

	if (calcule_dd) {
		SAFE_CUDA(cudaMalloc((void**)&ret->_locd2, sizeof(float) * (mdl->locd2s * sets * lines)));
		SAFE_CUDA(cudaMalloc((void**)&ret->_dd_weight, sizeof(float) * (dws * sets*ws)));
		SAFE_CUDA(cudaMalloc((void**)&ret->_dd_var, sizeof(float) * (mdl->total*sets*lines)));
		SAFE_CUDA(cudaMalloc((void**)&ret->_dd_locd, sizeof(float) * (mdl->locds*sets*lines)));
		SAFE_CUDA(cudaMalloc((void**)&ret->_dd_grad, sizeof(float) * (mdl->total*sets*lines)));
		SAFE_CUDA(cudaMalloc((void**)&ret->_dd_meand, sizeof(float) * (ws*sets)));
	}

	if (!calcule_d) ret->_locd = 0;
	
	if (!calcule_dd) {
		ret->_locd2 = 0;
		ret->_dd_weight = 0;
		ret->_dd_var = 0;
		ret->_dd_grad = 0;
		ret->_dd_locd = 0;
		ret->_dd_meand = 0;
	}

	//
	ret->set_score = (float*)malloc(sizeof(float) * sets);
	SAFE_CUDA(cudaMalloc((void**)&ret->set_score_d, sizeof(float) * sets));
	
	ret->set_rank = (uint*)malloc(sizeof(uint) * sets);
	SAFE_CUDA(cudaMalloc((void**)&ret->set_rank_d, sizeof(uint) * sets));

	ret->podium = (uint*)malloc(sizeof(uint) * sets);

	return ret;
};

Config_t * score_mk_config(uint id) {
	return config_mk(id, SCORE_params[id], SCORE_defaults[id]);
};

Config_t * opti_mk_config(uint id) {
	return config_mk(id, OPTI_params[id], OPTI_defaults[id]);
};

Config_t * gtic_mk_config(uint id) {
	return config_mk(id, GTIC_params[id], GTIC_defaults[id]);
};

void train_free(Train_t * ret) {
	free(ret->dw);
	
	SAFE_CUDA(cudaFree(ret->_weight));
	SAFE_CUDA(cudaFree(ret->_var));
	
	if (ret->calcule_d == 1) SAFE_CUDA(cudaFree(ret->_locd));

	SAFE_CUDA(cudaFree(ret->_grad));
	SAFE_CUDA(cudaFree(ret->_meand));
	
	if (ret->calcule_dd == 1) {
		SAFE_CUDA(cudaFree(ret->_locd2));
		SAFE_CUDA(cudaFree(ret->_dd_weight));
		SAFE_CUDA(cudaFree(ret->_dd_var));
		SAFE_CUDA(cudaFree(ret->_dd_locd));
		SAFE_CUDA(cudaFree(ret->_dd_grad));
		SAFE_CUDA(cudaFree(ret->_dd_meand));
	}

	SAFE_CUDA(cudaFree(ret->set_score_d));
	SAFE_CUDA(cudaFree(ret->set_rank_d));
	free(ret->set_rank);
	free(ret->set_score);
	free(ret->podium);
	
	GTIC_FREE[ret->gtic->id](ret);
	OPTI_FREE[ret->opti->id](ret);
	SCORE_FREE[ret->score->id](ret);

	config_free(ret->score);
	config_free(ret->opti);
	config_free(ret->gtic);

	free(ret);
};