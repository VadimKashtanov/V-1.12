#include "kernel/head/use.cuh"

void use_print_weights(Use_t * use) {
	float * tmpt = (float*)malloc(sizeof(float) * (use->mdl->weights));
	SAFE_CUDA(cudaMemcpy(tmpt, use->weight_d, sizeof(float) * use->mdl->weights, cudaMemcpyDeviceToHost));

	int lbl;
	for (uint i=0; i < use->mdl->weights; i++) {
		lbl = find_sep(use->mdl->wsep, i);

		if (lbl != -1)
			printf("|| (%i) %s\n", i, use->mdl->wsep->labels[lbl]);

		printf("|| %i |  \033[93m %f \033[0m \n", i, tmpt[i]);
	}

	free(tmpt);
};

void use_print_vars(Use_t * use) {
	float * tmpt = (float*)malloc(sizeof(float) * (use->data->lines * use->mdl->total));
	SAFE_CUDA(cudaMemcpy(tmpt, use->var_d, sizeof(float) * (use->data->lines * use->mdl->total), cudaMemcpyDeviceToHost));

	int lbl;
	uint pos;

	for (uint l=0; l < use->data->lines; l++) {
		printf("\033[%im||\033[0m", (l % 2 ? 92 : 91));
		printf("Line = %i ################### \n", l);
		for (uint i=0; i < use->mdl->total; i++) {
			lbl = find_sep(use->mdl->vsep, i);

			if (lbl != -1) {
				printf("\033[%im||\033[0m", (l % 2 ? 92 : 91));	// '||' de la ligne
				printf("|| (%i) %s\n", i, use->mdl->vsep->labels[lbl]);
			}

			pos = l*use->mdl->total + i;
				
			printf("\033[%im||\033[0m", (l % 2 ? 92 : 91));	// '||' de la ligne
				
			printf("|| %i |  \033[93m %f \033[0m \n", i, tmpt[pos]);
		}
	}

	free(tmpt);
};

//------------------

void use_print_all(Use_t * use) {
	use_print_weights(use);
	use_print_vars(use);
};
