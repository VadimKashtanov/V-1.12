#include "kernel/head/mdl.cuh"

void inst_print(Config_t * inst) {
	printf("[%s]\n", INST_name[inst->id]);
	for (uint i=0; i < inst->params; i++) {
		printf("%i| %s = %i\n", i, INST_param_name[inst->id][i], inst->param[i]);
	}
};

//=====================================================================

void mdl_print_inst(Mdl_t * mdl, uint i) {
	uint inst_id = mdl->inst[i]->id;
	printf("\033[30;1;46m %s\033[0m:\n", INST_name[inst_id]);
	for (uint p=0; p < INST_params[inst_id]; p++)
		printf("\t\033[42;1;42m%s\033[0m: %i\n", INST_param_name[inst_id][p], mdl->inst[i]->param[p]);
};

void mdl_print_insts(Mdl_t * mdl) {
	for (uint i=0; i < mdl->insts; i++)
		mdl_print_inst(mdl, i);
};

//=========================================================================

void mdl_print_vseps(Mdl_t * mdl) {
	printf("	Variables Separators\n");
	printf("   labels      positions\n");
	for (uint i=0; i < mdl->vsep->seps; i++)
		printf("%i| %s   %i\n", i, mdl->vsep->labels[i], mdl->vsep->sep_pos[i]);
};

void mdl_print_wseps(Mdl_t * mdl) {
	printf("	Weights Separators\n");
	printf("   labels      positions\n");
	for (uint i=0; i < mdl->wsep->seps; i++)
		printf("%i| %s   %i\n", i, mdl->wsep->labels[i], mdl->wsep->sep_pos[i]);
};

void mdl_print_lseps(Mdl_t * mdl) {
	printf("	Locds Separators\n");
	printf("   labels      positions\n");
	for (uint i=0; i < mdl->lsep->seps; i++)
		printf("%i| %s   %i\n", i, mdl->lsep->labels[i], mdl->lsep->sep_pos[i]);
};

void mdl_print_l2seps(Mdl_t * mdl) {
	printf("	Locd2s Separators\n");
	printf("   labels      positions\n");
	for (uint i=0; i < mdl->l2sep->seps; i++)
		printf("%i| %s   %i\n", i, mdl->l2sep->labels[i], mdl->l2sep->sep_pos[i]);
};

//=========================================================================

void mdl_print_weights(Mdl_t * mdl) {
	int lbl;
	for (uint i=0; i < mdl->weights; i++) {
		lbl = find_sep(mdl->wsep, i);

		if (lbl != -1)
			printf("|| (%i) %s\n", i, mdl->wsep->labels[lbl]);

		printf("|| %i |  \033[93m %f \033[0m \n", i, mdl->weight[i]);
	}
};

void mdl_compare_weights(Mdl_t * mdl, float * with_this) {
	printf("	mdl->weight   compare_array\n");
	int lbl;
	for (uint i=0; i < mdl->weights; i++) {
		lbl = find_sep(mdl->wsep, i);

		if (lbl != -1)
			printf("|| (%i) %s\n", i, mdl->wsep->labels[lbl]);

		if (compare_floats(mdl->weight[i], with_this[i], 0.001)) 
			printf("|| %i |  \033[42m %f --- %f \033[0m \n", i, mdl->weight[i], with_this[i]);
		else
			printf("|| %i |  \033[41m %f --- %f \033[0m \n", i, mdl->weight[i], with_this[i]);
	}

	printf("             C/Cuda  ||| Python\n");
};