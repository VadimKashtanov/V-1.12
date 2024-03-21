#include <stdio.h>
#include "package/package.cuh"

//./print_mdl mdl.bin yes
//	"yes" will print mdl with all weights
//	anything else than "yes" will no print
//./print_mdl mdl.bin no
//	will not print

int main(int argc, const char ** argv) {
	if (argc != 3)
		ERR("You have to give 1 file and if you want or not (yes/no) to show weights")

	FILE * fp = fopen(argv[1], "rb");

	Mdl_t * mdl = mdl_fp_load(fp);
	mdl_check_correctness(mdl);

	printf(" ---- Instructions ----\n");
	mdl_print_insts(mdl);

	printf(" ---- Separateurs -----\n");
	mdl_print_vseps(mdl);
	mdl_print_wseps(mdl);
	mdl_print_lseps(mdl);
	mdl_print_l2seps(mdl);

	printf(" ------- Valeurs ------ \n");
	printf("	Inputs  = %i\n", mdl->inputs);
	printf("	Outputs = %i\n", mdl->outputs);
	printf("	Total   = %i\n", mdl->total);
	printf("	Vars    = %i\n", mdl->vars);
	printf("	Weights = %i\n", mdl->weights);
	printf("	Locds   = %i\n", mdl->locds);
	printf("	Locd2s  = %i\n", mdl->locds);

	if (strcmp(argv[2], "yes") == 0)
		mdl_print_weights(mdl);

	mdl_free(mdl);
	fclose(fp);
};