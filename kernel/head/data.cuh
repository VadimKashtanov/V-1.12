#pragma once

#include "kernel/head/etc.cuh"

/*	The Data struct store 1 batch (of `lines` lines).
It can be load from a binary file with multiple batchs.
It can be build juste for storing input for Cpu or Use. 	*/

typedef struct data_struct {
	//	Params
	uint batchs;
	uint inputs, outputs;
	uint lines;				//batchs is relative to a Data file. It can contain the same batch but more or less other batchs

	//	Load here the fread() array
	float * input, * output;	//[lines][inps/outs]

	//	Nvidia Vram
	float * input_d, * output_d;
} Data_t;

//	Mem
Data_t * data_open(char * file);
Data_t * data_load(uint batchs, uint inputs, uint outputs, uint lines);
void data_cudamalloc(Data_t * data);

//	Controle
void data_load_input_batch(Data_t * data, FILE * fp);
void data_load_output_batch(Data_t * data, FILE * fp);

void data_open_batch(Data_t * data, char * file, uint batch);
void data_load_batch(Data_t * data, FILE * fp, uint batch);
void data_cudamemcpy(Data_t * data);

//	Plum
void data_print_info(Data_t * data);
void data_print_batch(Data_t * data);

//	Free
void data_free_cudamalloc(Data_t * data);
void data_free(Data_t * data);

/*	Data binary file Structure
uint 	batchs 	x1
uint 	lines 	x1
uint 	inputs	x1
uint 	outputs	x1
[batchs]
	[lines]
		float 	input 	xinputs
	[end]
[end]
[batchs]
	[lines]
		float	output 	xoutputs
	[end]
[end]
*/
