#include "kernel/head/analyse/hessienne.cuh"

/*
	M      1/M
| 1 2 3 | 1 0 0 |
| 4 5 6 | 0 1 0 |
| 7 8 9 | 0 0 1 |

Pivot de gauss

L1 = L1 - L0*4
...
Jusqu'a avoire Id a la place de M

*/

bool CPU_invert_hessienne(Hessienne_t * hessienne) {
	Train_t * train = hessienne->opti->train;
	Mdl_t * mdl = train->mdl;

	uint wsize = mdl->weights;

	uint n = wsize;

	float a[n*n];

	//
	for (uint set=0; set < train->sets; set++) {

		//	Id[n] pour l'inverse
		for (uint i=0; i < n; i++) {
			for (uint j=0; j < n; j++) {
				hessienne->inverse_par_set[set*wsize*wsize + i*n + j] = ( i == j ? 1 : 0);
				a[i*n + j] = hessienne->tableau[set*wsize*wsize + i*n + j];
			}
		}

		//On fait le pivot
		float coef;
		for (uint L=0; L < n; L++) {
			for (uint y=0; y < n; y++) {
				if (y == L) continue;

				if (a[L*n + L] == 0) {
					MSG("Non inversible");
					return false;
				}

				//Ly -= LL * coef
				coef = a[y*n + L] / a[L*n + L];
				for (uint k=0; k < n; k++) {
					a[y*n + k] -= a[L*n + k]*coef;
					hessienne->inverse_par_set[set*wsize*wsize + y*n + k] -= hessienne->inverse_par_set[set*wsize*wsize + L*n + k]*coef;
				}
			}
		};

		for (uint L=0; L < n; L++) {
			coef = a[L*n + L];
			if (coef == 0) {
				MSG("Non inversible");
				return false;
			}
			for (uint i=0; i < n; i++) {
				a[L*n + i] /= coef;
				hessienne->inverse_par_set[set*wsize*wsize + L*n + i] /= coef;
			}
		}
	}

	SAFE_CUDA(cudaMemcpy(hessienne->inverse_par_set_d, hessienne->inverse_par_set, sizeof(float) * train->sets * wsize * wsize, cudaMemcpyHostToDevice));

	return true;
};