#include "package/optis/optis.cuh"

//=========================== Des Constantes implémenté ici car plus simple ==========================

static const char* sgd_params_names[1] = {
	"alpha"
};

static const uint sgd_params_defaults[1] = {
	1036831949
};

//========================== Toutes les arrays de fonctions ou de constantes =========================

const char* OPTI_name[OPTIS] = {
	"sgd"
};

const uint OPTI_params[OPTIS] = {
	1, //SGD
};

const char** OPTI_params_names[OPTIS] = {
	sgd_params_names,	//sgd
};

const uint* OPTI_defaults[OPTIS] = {
	sgd_params_defaults,	//sgd
};

const uint OPTI_require_ddf[OPTIS] = {
	0, //SGD
};

dict_config_f OPTI_STR_CONFIG[OPTIS] = {
	sgd_str_config,	//sgd
};

func_train_f OPTI_MK[OPTIS] = {
	sgd_mk,	//sgd
};

func_train_f OPTI_OPTI[OPTIS] = {
	sgd_opti,	//sgd
};

func_train_f OPTI_FREE[OPTIS] = {
	sgd_free,	//sgd
};

