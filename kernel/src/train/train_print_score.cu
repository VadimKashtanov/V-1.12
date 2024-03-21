#include "kernel/head/train.cuh"

void score_print_scores(Train_t * train) {
	printf("| Set| score |\n");
	printf("|====|=======|\n");
	for (uint i=0; i < train->sets; i++) {
		printf("|%4.i|%.5g|\n", i, train->set_score[i]);
	}
};

void score_print_ranks(Train_t * train) {
	printf("| Set| rank  |\n");
	printf("|====|=======|\n");
	for (uint i=0; i < train->sets; i++) {
		printf("|%4.i|%5.i|\n", i, train->set_rank[i]);
	}
};

void score_print_podium(Train_t * train) {
	printf("| Rank | Set id  |\n");
	printf("|======|=========|\n");
	for (uint i=0; i < train->sets; i++) {
		printf("|%4.i|%5.i|\n", i, train->podium[i]);
	}
};

uint score_eq_score(Train_t * train, float * compare, float tolerance) {
	for (uint i=0; i < train->sets; i++)
		if (compare_floats(train->set_score[i], compare[i], tolerance) == 0)
			return 0;
	return 1;
};

void score_compare_scores(Train_t * train, float * compare, float tolerance) {
	printf("============= Comparaison des Scores ============\n");
	for (uint i=0; i < train->sets; i++) {
		if (compare_floats(train->set_score[i], compare[i], tolerance) == 1) {
			printf("%i| \033[42m%.4g ; %.4g\033[0m\n", i, train->set_score[i], compare[i]);
		} else {
			printf("%i| \033[101m%.4g ; %.4g\033[0m\n", i, train->set_score[i], compare[i]);
		}
	}
	printf("Train->set_score |  compare\n");
};