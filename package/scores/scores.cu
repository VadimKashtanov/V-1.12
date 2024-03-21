#include "package/scores/scores.cuh"

//=========================== Des Constantes implémenté ici car plus simple ==========================

static const char* meansquared_params_names[0] = {
	
};

static const uint meansquared_params_defaults[0] = {
	
};

//========================== Toutes les arrays de fonctions ou de constantes =========================

const char* SCORE_name[SCORES] = {
	"meansquared"
};

const uint SCORE_params[SCORES] = {
	0, //MEANSQUARED
};

const char** SCORE_params_names[SCORES] = {
	meansquared_params_names,	//meansquared
};

const uint* SCORE_defaults[SCORES] = {
	meansquared_params_defaults,	//meansquared
};

const uint SCORE_allow_ddf[SCORES] = {
	1, //MEANSQUARED
};

dict_config_f SCORE_STR_CONFIG[SCORES] = {
	meansquared_str_config,	//meansquared
};

func_train_f SCORE_MK[SCORES] = {
	meansquared_mk,	//meansquared
};

func_train_f SCORE_SCORE[SCORES] = {
	meansquared_score,	//meansquared
};

func_train_f SCORE_LOSS[SCORES] = {
	meansquared_loss,	//meansquared
};

func_train_f SCORE_DLOSS[SCORES] = {
	meansquared_dloss,	//meansquared
};

func_train_f SCORE_DDLOSS[SCORES] = {
	meansquared_ddloss,	//meansquared
};

func_train_f SCORE_FREE[SCORES] = {
	meansquared_free,	//meansquared
};

