#include "kernel/head/data.cuh"

void data_print_info(Data_t * data) {
	printf("\033[96m Data_t \033[0m\n");
	printf("\033[93m Batchs : %i \033[0m\n", data->batchs);
	printf("\033[93m Lines : %i \033[0m\n", data->lines);
	printf("\033[93m Inputs : %i \033[0m\n", data->inputs);
	printf("\033[93m Outputs : %i \033[0m\n", data->outputs);
	printf("\033[93m Cuda Loaded : %i \033[0m\n", (int)((data->input_d!=0) * (data->output_d!=0)));
};

void data_print_batch(Data_t * data) {
	printf("\033[100m ########## Input ########### \033[0m \n");
	for (uint l=0; l < data->lines; l++) {
		printf("\033[92m ==== Line %i ==== \033[0m \n", l);
		for (uint i=0; i < data->inputs; i++)
			printf("\033[94m%f\033[0m,", data->input[l*data->inputs + i]);
		printf("\n");
	}

	printf("\033[100m ########## Output ########### \033[0m \n");
	for (uint l=0; l < data->lines; l++) {
		printf("\033[92m ==== Line %i ==== \033[0m \n", l);
		for (uint i=0; i < data->outputs; i++)
			printf("\033[94m%f\033[0m,", data->output[l*data->outputs + i]);
		printf("\n");
	}
};