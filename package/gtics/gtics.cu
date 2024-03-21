#include "package/gtics/gtics.cuh"

//=========================== Des Constantes implémenté ici car plus simple ==========================

static const char* neutre_params_names[1] = {
	"sets"
};

static const uint neutre_params_defaults[1] = {
	1
};

//========================== Toutes les arrays de fonctions ou de constantes =========================

const char* GTIC_name[GTICS] = {
	"neutre"
};

const uint GTIC_params[GTICS] = {
	1, //NEUTRE
};

const char** GTIC_params_names[GTICS] = {
	neutre_params_names,	//neutre
};

const uint* GTIC_defaults[GTICS] = {
	neutre_params_defaults,	//neutre
};

dict_config_f GTIC_STR_CONFIG[GTICS] = {
	neutre_str_config,	//neutre
};

func_train_f GTIC_MK[GTICS] = {
	neutre_mk,	//neutre
};

func_train_f GTIC_GTIC[GTICS] = {
	neutre_gtic,	//neutre
};

func_train_f GTIC_FREE[GTICS] = {
	neutre_free,	//neutre
};

