#include "kernel/head/mdl.cuh"

void mdl_free(Mdl_t * mdl) {
	//	Separators
	sep_free(mdl->vsep);
	sep_free(mdl->wsep);
	sep_free(mdl->lsep);
	sep_free(mdl->l2sep);
	
	//	Insts
	for (uint i=0; i < mdl->insts; i++)
		config_free(mdl->inst[i]);

	//Ws
	free(mdl->weight);

	//
	free(mdl);
};
