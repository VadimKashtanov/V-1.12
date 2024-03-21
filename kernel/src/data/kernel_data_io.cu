#include "kernel/head/data.cuh"

Data_t * data_open(char * file) {
	FILE * fp = SAFE_FOPEN(file, "rb");

	uint batchs, lines, inputs, outputs;

	fread(&batchs, sizeof(uint), 1, fp);
	fread(&lines, sizeof(uint), 1, fp);
	fread(&inputs, sizeof(uint), 1, fp);
	fread(&outputs, sizeof(uint), 1, fp);

	fclose(fp);

	return data_load(batchs, inputs, outputs, lines);
};

void data_load_batch(Data_t * data, FILE * fp, uint batch) {
	//	Seek to input `batch` batch
	fseek(fp,
		sizeof(uint)*4 + sizeof(float)*(batch * data->lines*data->inputs),
		SEEK_SET);
    data_load_input_batch(data, fp);
	
	//	Seek to output `batch` batch
	fseek(fp,
		sizeof(uint)*4 + sizeof(float)*(data->batchs*data->lines*data->inputs + batch*data->lines*data->outputs),
		SEEK_SET);
    data_load_output_batch(data, fp);
};

void data_load_input_batch(Data_t * data, FILE * fp) {
    fread(data->input, sizeof(float), data->lines*data->inputs, fp);
};

void data_load_output_batch(Data_t * data, FILE * fp) {
    fread(data->output, sizeof(float), data->lines*data->outputs, fp);
};

