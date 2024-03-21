#include "package/package.cuh"

//./mdl_weight_limits mdl.bin -3 3
//		ca va juste faire min(max(-3, weight), 3)

int main(int argc, char ** argv) {
	if (argc != 4)
		ERR("You have to give : mdl, borne0, borne1");

	FILE * fp = fopen(argv[1], "rb");
	Mdl_t * mdl = mdl_fp_load(fp);
	mdl_check_correctness(mdl);
	fclose(fp);

	float borne0 = atof(argv[2]), borne1 = atof(argv[3]);

	for (uint i=0; i < mdl->weights; i++) {
		if (mdl->weight[i] > borne1) mdl->weight[i] = borne1;
		if (mdl->weight[i] < borne0) mdl->weight[i] = borne0;
	};

	fp = fopen(argv[1], "wb");
	mdl_fp_write(mdl, fp);
	fclose(fp);
};