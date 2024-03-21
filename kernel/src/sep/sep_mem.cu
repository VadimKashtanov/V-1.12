#include "kernel/head/sep.cuh"

int find_sep(Separators_t * sep, uint this_pixel) {
	for (uint i=0; i < sep->seps; i++)
		if (sep->sep_pos[i] == this_pixel)
			return i;
	return -1;
};

void sep_free(Separators_t * sep) {
	for (uint i=0; i < sep->seps; i++) {
		free(sep->labels[i]);
	}
	free(sep->labels);
	//
	free(sep->sep_pos);
	free(sep);
};