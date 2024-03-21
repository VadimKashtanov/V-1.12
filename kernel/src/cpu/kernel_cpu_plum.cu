#include "kernel/head/cpu.cuh"

void cpu_print_vars(Cpu_t * cpu) {
	int lbl;
	uint pos;
	
	for (uint l=0; l < cpu->data->lines; l++) {
		printf("\033[%im||\033[0m", (l % 2 ? 92 : 91));
		printf("Line = %i ################### \n", l);
		for (uint i=0; i < cpu->mdl->total; i++) {
			lbl = find_sep(cpu->mdl->vsep, i);
			
			if (lbl != -1) {
				printf("\033[%im||\033[0m", (l % 2 ? 92 : 91));	// '||' de la ligne
				printf("|| (%i) %s\n", i, cpu->mdl->vsep->labels[lbl]);
			}
			
			pos = l*cpu->mdl->total + i;
			
			printf("\033[%im||\033[0m", (l % 2 ? 92 : 91));	// '||' de la ligne
			
			printf("|| %i |  \033[93m %f \033[0m \n", i, cpu->var[pos]);
		}
	}
};

void cpu_compare_vars(Cpu_t * cpu, float * with_this) {
	int lbl;
	uint pos;

	for (uint l=0; l < cpu->data->lines; l++) {
		printf("\033[%im||\033[0m", (l % 2 ? 92 : 91));
		printf("Line = %i ################### \n", l);
		for (uint i=0; i < cpu->mdl->total; i++) {
			lbl = find_sep(cpu->mdl->vsep, i);

			if (lbl != -1) {
				printf("\033[%im||\033[0m", (l % 2 ? 92 : 91));	// '||' de la ligne
				printf("|| (%i) %s\n", i, cpu->mdl->vsep->labels[lbl]);
			}

			pos = l*cpu->mdl->total + i;
			
			printf("\033[%im||\033[0m", (l % 2 ? 92 : 91));	// '||' de la ligne
			
			if (compare_floats(cpu->var[pos], with_this[pos], 0.001)) 
				printf("|| %i |  \033[42m %f --- %f \033[0m \n", i, cpu->var[pos], with_this[pos]);
			else
				printf("|| %i |  \033[41m %f --- %f \033[0m \n", i, cpu->var[pos], with_this[pos]);
		}
	}

	printf("             C/Cuda  ||| Python\n");
};
