#include "package/insts/insts.cuh"

//=========================== Des Constantes implémenté ici car plus simple ==========================

static const char* dot1d_params_names[8] = {
	"Ax", "Yx", "activ", "istart", "ystart", "wstart", "lstart", "l2start"
};

//========================== Toutes les arrays de fonctions ou de constantes =========================

uint INST_params[INSTS] = {
	8, //DOT1D
};

const char* INST_name[INSTS] = {
	"dot1d"
};

const char** INST_param_name[INSTS] = {
	dot1d_params_names,	//dot1d
};

uint INST_capable_df[INSTS] = {
	1, //dot1d
};

uint INST_capable_ddf[INSTS] = {
	1, //dot1d
};

inst_check_f INST_CHECK[INSTS] = {
	dot1d_check,	//dot1d
};

cpu_f INST_CPU[INSTS] = {
	dot1d_cpu,	//dot1d
};

use_f INST_USE[INSTS] = {
	dot1d_use,	//dot1d
};

train_f INST_FORWARD[INSTS] = {
	dot1d_forward,	//dot1d
};

train_f INST_BACKWARD[INSTS] = {
	dot1d_backward,	//dot1d
};

train_f INST_FORWARD2[INSTS] = {
	dot1d_forward2,	//dot1d
};

train_f INST_BACKWARD2[INSTS] = {
	dot1d_backward2,	//dot1d
};

train_dw_f INST_BACKWARD_OF_BACKWARD2[INSTS] = {
	dot1d_backward_of_backward2,	//dot1d
};

train_dw_f INST_BACKWARD_OF_FORWARD2[INSTS] = {
	dot1d_backward_of_forward2,	//dot1d
};

