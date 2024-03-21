#include "kernel/head/analyse/hessienne.cuh"

bool opti_tout_hessienne(Hessienne_t * hessienne) {
	calculer_tableau_hessienne(hessienne);
	if (CPU_invert_hessienne(hessienne) == false)
		return false;
	opti_hessienne(hessienne);
	return true;
};

void hessienne_print_matrice(Hessienne_t * hessienne) {
	Opti_t * opti = hessienne->opti;
	Train_t * train = opti->train;
	Mdl_t * mdl = train->mdl;

	uint wsize = mdl->weights;
	uint sets = train->sets;

	uint n = wsize;

	train_set_input(train);

	float valeur;
	uint couleur;

	for (uint set=0; set < sets; set++) {
		printf("\033[93m ============ SET #%i ============ \033[0m\n", set);
		for (uint i=0; i < n; i++) {
			for (uint j=0; j < n; j++) {
				valeur = hessienne->tableau[set*wsize*wsize + i*n + j];

				couleur = ((valeur < 1 && valeur > -1) ? 94 : 92);	//c'est un nombre entre 1;-1 donc peut en 0.xxx ou -0.xxx

				if (i==j)
					printf("\033[100;%im%.5e\033[0m, ", couleur, valeur);
				else
					printf("\033[%im%.5e\033[0m, ", couleur, valeur);
			}
			printf("\n");
		}
	}
}