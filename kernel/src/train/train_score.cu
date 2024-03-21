#include "kernel/head/train.cuh"

void train_solo_compute_score(Train_t * train, uint start_seed) {
	//
	train_set_input(train);
	train_forward(train, start_seed);

	//	Loss
	score_loss(train);
	score_score(train);
};

static void ordoner_les_scores(Train_t * train) {
	uint c;
	for (uint i=0; i < train->sets; i++) train->set_rank[i] = i;
	for (uint i=0; i < train->sets; i++) {
		for (uint j=0; j < train->sets; j++) {
			if (train->set_score[train->set_rank[i]] < train->set_score[train->set_rank[j]]) {
				c = train->set_rank[i];
				train->set_rank[i] = train->set_rank[j];
				train->set_rank[j] = c;
			}
		}
	}

	SAFE_CUDA(cudaMemcpy(train->set_rank_d, train->set_rank, sizeof(uint)*train->sets, cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(train->set_score_d, train->set_score, sizeof(float)*train->sets, cudaMemcpyHostToDevice));
}

void score_score(Train_t * train) {
	SCORE_SCORE[train->score->id](train);	//calcule train->set_score a partire de loss (il faut obligatoirement faire loss avant)

	ordoner_les_scores(train);
};

void score_loss(Train_t * train) {
	SCORE_LOSS[train->score->id](train);
};

void score_dloss(Train_t * train) {
	SCORE_DLOSS[train->score->id](train);
};

void score_ddloss(Train_t * train) {
	SCORE_DDLOSS[train->score->id](train);
};