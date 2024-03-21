#pragma once

#include "../train.cuh"

typedef struct {
	//	Le tableau des dérivés secondes
	//float * tableau_d;	//[set*wsize*wsize + w0*wsize + w1] //d/dw0dw1
	float * tableau_d;
	float * tableau;

	//	Inverses
	float * inverse_par_set_d;	//[set*wsize*wsize + y*wsize + x]
	float * inverse_par_set;
} Hessienne_t;

//	Hessienne
Hessienne_t * mk_hessienne(Opti_t * opti);
void cudmalloc_hessienne(Hessienne_t * hessienne);

//	Calcule procedurale du tables
void calculer_tableau_hessienne(Hessienne_t * hessienne);

//	Inversion matrice
bool CPU_invert_hessienne(Hessienne_t * hessienne);
bool GPU_invert_hessienne(Hessienne_t * hessienne);

//	Optimisation
void opti_hessienne(Hessienne_t * hessienne);

//	Automatique
bool opti_tout_hessienne(Hessienne_t * hessienne); //calcule table, puis inversion, puis optimisation a chaque fois
void hessienne_print_matrice(Hessienne_t * hessienne);

//Free
void free_hessienne(Hessienne_t * hessienne);