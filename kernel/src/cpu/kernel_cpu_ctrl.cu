#include "kernel/head/cpu.cuh"

void cpu_set_input(Cpu_t * cpu) {
	Data_t * data = cpu->data;
	
	for (uint t=0; t < cpu->data->lines; t++)
		memcpy(cpu->var + t*cpu->mdl->total, data->input + t*cpu->mdl->inputs, sizeof(float) * cpu->mdl->inputs);
};

void cpu_forward(Cpu_t * cpu) {
	for (uint t=0; t < cpu->data->lines; t++)
		for (uint i=0; i < cpu->mdl->insts; i++)
			INST_CPU[cpu->mdl->inst[i]->id](cpu, cpu->mdl->inst[i], t);
};