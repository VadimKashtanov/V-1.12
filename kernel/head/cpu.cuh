#pragma once

#include "kernel/head/mdl.cuh"
#include "kernel/head/data.cuh"

typedef struct cpu_model_forward {
	//	Model where the *weight is stored
	Mdl_t * mdl;

	//	Data where the input is stored. The output will be set to 0 because it's useless. Data is build without `data_load`
	Data_t * data;

	//times == data->lines;
	float * var;
} Cpu_t;

//	Mem
Cpu_t* cpu_mk(Mdl_t * mdl, Data_t * data);
void cpu_free(Cpu_t * cpu);

//	Ctrl
void cpu_set_input(Cpu_t * cpu);
void cpu_forward(Cpu_t * cpu);

//	Plum
void cpu_print_vars(Cpu_t * cpu);
//
void cpu_compare_vars(Cpu_t * cpu, float * with_this);

//	List of instructions executed on cpu
typedef void (*cpu_f)(Cpu_t* cpu, Config_t * inst, uint time);
extern cpu_f INST_CPU[INSTS];