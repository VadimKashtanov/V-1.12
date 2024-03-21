#pragma once

#include "kernel/head/etc.cuh"

typedef struct {
	uint seps;
	uint * sep_pos;	//position in 1 line
	char ** labels;	//label of the separator
} Separators_t;

//	Sep I/O
Separators_t * sep_fp_load(FILE * fp);
void sep_fp_write(FILE * fp, Separators_t * sep);

//	Sep controle
int find_sep(Separators_t * sep, uint this_pixel);

//	Sep mem
void sep_free(Separators_t * sep);
