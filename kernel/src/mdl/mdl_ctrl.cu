#include "kernel/head/mdl.cuh"

void inst_check(Config_t * inst) {
	if (inst->id >= INSTS)
		ERR("inst->id = %i. Max id = %i", inst->id, INSTS-1);

	if (inst->params > INST_params[inst->id])
		ERR("inst->params = %i. Inst %s a %i parametres", inst->params, INST_name[inst->id], INST_params[inst->id]);

	INST_CHECK[inst->id](inst);
};

void mdl_check_correctness(Mdl_t * mdl) {
	//	Pour l'instant on met une limite d'instructions de 1000
	if (mdl->insts > MAX_INSTS)
		raise(SIGINT);

	//	On check si les informations sur les instructions sont coherantes
	for (uint i=0; i < mdl->insts; i++) {
		inst_check(mdl->inst[i]);
	}

	//	On verifie que mdl->total est bien calculé
	if (mdl->total != mdl->inputs + mdl->vars)
		raise(SIGINT);

	//	On verifie que les Separateur sont pas incoherants
	//vsep
	for (uint i=0; i < mdl->vsep->seps; i++)
		if (mdl->vsep->sep_pos[i] >= mdl->total)
			raise(SIGINT);	//la position ne peut pas etre apres la ligne.
	//wsep
	for (uint i=0; i < mdl->wsep->seps; i++)
		if (mdl->wsep->sep_pos[i] >= mdl->weights)
			raise(SIGINT);	//la position ne peut pas etre apres la ligne.
	//lsep
	for (uint i=0; i < mdl->lsep->seps; i++)
		if (mdl->lsep->sep_pos[i] >= mdl->locds)
			raise(SIGINT);	//la position ne peut pas etre apres la ligne.
	//l2sep
	for (uint i=0; i < mdl->l2sep->seps; i++)
		if (mdl->l2sep->sep_pos[i] >= mdl->locd2s)
			raise(SIGINT);	//la position ne peut pas etre apres la ligne.
};