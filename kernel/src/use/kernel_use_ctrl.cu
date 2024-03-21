#include "kernel/head/use.cuh"

void use_set_input(Use_t * use) {
	for (uint t=0; t < use->data->lines; t++) {
		SAFE_CUDA(
			cudaMemcpy(
				use->var_d + t*use->mdl->total,
				use->data->input_d + t*use->mdl->inputs,
				sizeof(float) * use->mdl->inputs,
				cudaMemcpyDeviceToDevice
			)
		)
	}
};

void use_forward(Use_t * use) {
	for (uint t=0; t < use->data->lines; t++)
		for (uint i=0; i < use->mdl->insts; i++)
			INST_USE[use->mdl->inst[i]->id](use, use->mdl->inst[i], t);
};