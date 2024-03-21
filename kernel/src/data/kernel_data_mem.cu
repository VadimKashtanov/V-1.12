#include "kernel/head/data.cuh"

Data_t * data_load(uint batchs, uint inputs, uint outputs, uint lines) {
	Data_t * ret = (Data_t*)malloc(sizeof(Data_t));

	ret->batchs = batchs;
	ret->inputs = inputs;
	ret->outputs = outputs;
	ret->lines = lines;

	ret->input = (float*)malloc(sizeof(float) * ret->lines * ret->inputs);
	ret->output = (float*)malloc(sizeof(float) * ret->lines * ret->outputs);

	ret->input_d = 0;
	ret->output_d = 0;

	return ret;
};

void data_cudamalloc(Data_t * data) {
	SAFE_CUDA(cudaMalloc((void**)&data->input_d, sizeof(float) * data->lines * data->inputs));
	SAFE_CUDA(cudaMalloc((void**)&data->output_d, sizeof(float) * data->lines * data->outputs));
};

void data_cudamemcpy(Data_t * data) {
	SAFE_CUDA(cudaMemcpy(
		data->input_d,
		data->input,
		sizeof(float) * data->inputs * data->lines,
		cudaMemcpyHostToDevice))

	SAFE_CUDA(cudaMemcpy(
		data->output_d,
		data->output,
		sizeof(float) * data->outputs * data->lines,
		cudaMemcpyHostToDevice))
};

void data_free_cudamalloc(Data_t * data) {
	if (data->input_d) SAFE_CUDA(cudaFree(data->input_d));
	if (data->output_d) SAFE_CUDA(cudaFree(data->output_d));
};

void data_free(Data_t * data) {
	if (data->input) free(data->input);
	if (data->output) free(data->output);

	data_free_cudamalloc(data);

	free(data);
};
