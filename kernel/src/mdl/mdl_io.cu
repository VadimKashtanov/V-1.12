#include "kernel/head/mdl.cuh"

Mdl_t* mdl_fp_load(FILE * fp) {
	Mdl_t * ret = (Mdl_t*)malloc(sizeof(Mdl_t));

	//fread(uint)
	is_123(fp);

	/*			Instructions		*/
	fread(&ret->insts, sizeof(uint), 1, fp);

	ret->inst = (Config_t**)malloc(sizeof(Config_t*) * ret->insts);

	//la prochaine boucle calcule la valeur
	ret->capable_df = 1;
	ret->capable_ddf = 1;

	for (uint i=0; i < ret->insts; i++) {
		ret->inst[i] = config_load(fp);
		inst_check(ret->inst[i]);

		if (INST_capable_df[ret->inst[i]->id] == 0) ret->capable_df = 0;
		if (INST_capable_ddf[ret->inst[i]->id] == 0) ret->capable_ddf = 0;
	}

	fread(&ret->inputs, sizeof(uint), 1, fp);
	fread(&ret->outputs, sizeof(uint), 1, fp);

	fread(&ret->vars, sizeof(uint), 1, fp);
	fread(&ret->weights, sizeof(uint), 1, fp);
	fread(&ret->locds, sizeof(uint), 1, fp);
	fread(&ret->locd2s, sizeof(uint), 1, fp);

	ret->weight = (float*)malloc(sizeof(float) * ret->weights);
	
	fread(ret->weight, sizeof(float), ret->weights, fp);

	ret->total = ret->inputs + ret->vars;

	//	Separators
	ret->vsep = sep_fp_load(fp);
	ret->wsep = sep_fp_load(fp);
	ret->lsep = sep_fp_load(fp);
	ret->l2sep = sep_fp_load(fp);

	//fread(uint)
	//out_read_123(fp);
	
	return ret;
};

Mdl_t* mdl_open(char * file) {
	FILE * fp = SAFE_FOPEN(file, "rb");
	Mdl_t * ret = mdl_fp_load(fp);
	fclose(fp);
	return ret;
}

void mdl_fp_write(Mdl_t * mdl, FILE * fp) {
	write_123(fp);

	fwrite(&mdl->insts, sizeof(uint), 1, fp);

	for (uint i=0; i < mdl->insts; i++) {
		config_write(mdl->inst[i], fp);
	}

	fwrite(&mdl->inputs, sizeof(uint), 1, fp);
	fwrite(&mdl->outputs, sizeof(uint), 1, fp);

	fwrite(&mdl->vars, sizeof(uint), 1, fp);
	fwrite(&mdl->weights, sizeof(uint), 1, fp);
	fwrite(&mdl->locds, sizeof(uint), 1, fp);
	fwrite(&mdl->locd2s, sizeof(uint), 1, fp);

	fwrite(mdl->weight, sizeof(float), mdl->weights, fp);

	sep_fp_write(fp, mdl->vsep);
	sep_fp_write(fp, mdl->wsep);
	sep_fp_write(fp, mdl->lsep);
	sep_fp_write(fp, mdl->l2sep);
};