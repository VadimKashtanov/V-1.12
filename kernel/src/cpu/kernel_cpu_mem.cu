#include "kernel/head/cpu.cuh"

Cpu_t* cpu_mk(Mdl_t * mdl, Data_t * data) {
	Cpu_t * ret = (Cpu_t*)malloc(sizeof(Cpu_t));
	ret->mdl = mdl;
	ret->data = data;
	ret->var = (float*)malloc(sizeof(float) * data->lines * mdl->total);
	return ret;
};

void cpu_free(Cpu_t * cpu) {
	free(cpu->var);
	free(cpu);
};
