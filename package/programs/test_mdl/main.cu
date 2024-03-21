#include "package/package.cuh"

//./test_mdl mdl.bin data.bin 2 a
//		will cpu_t on batch=2 and will output all lines
//
//./test_mdl mdl.bin data.bin 2 8
//		will cpu_t on batch=2, will output line=8 only

int main(int argc, char ** argv) {
	if (argc != 5)
		ERR("You have to give : mdl, data, batch_id, line_id");

	FILE * fp;

	fp = fopen(argv[1], "rb");
	Mdl_t * mdl = mdl_fp_load(fp);
	mdl_check_correctness(mdl);
	fclose(fp);

	Data_t * data = data_open(argv[2]);
	fp = fopen(argv[2], "rb");
	data_load_batch(data, fp, atoi(argv[3]));
	fclose(fp);

	//	Use mdl & data
	Cpu_t * cpu = cpu_mk(mdl, data);
	cpu_set_input(cpu);
	cpu_forward(cpu);

	//cpu_print_vars(cpu);

	if (strcmp(argv[4], "a") == 0) {
		for (uint l=0; l < data->lines; l++) {
			printf(" -- \033[105mLine\033[0m %i --\n", l);
			for (uint o=0; o < data->outputs; o++)
				printf("\033[93m%.3g\033[0m,", cpu->var[l*mdl->total + (mdl->total-mdl->outputs) + o]);
			printf("\n");
		}
	} else {
		uint line = atoi(argv[4]);
		printf(" -- \033[105mLine\033[0m %i --  (batchs=%i, lines=%i)\n", line, data->batchs, data->lines);
		printf("Prédicted -> ");
		for (uint o=0; o < data->outputs; o++)
			printf("\033[93m%.3g\033[0m,", cpu->var[(line)*mdl->total + (mdl->total-mdl->outputs) + o]);
		printf("\n");
		printf("Real Data -> ");
		for (uint o=0; o < data->outputs; o++)
			printf("\033[93m%.3g\033[0m,", data->output[(line)*data->outputs + o]);
		printf("\n");
	}

	cpu_free(cpu);
	data_free(data);
	mdl_free(mdl);
};