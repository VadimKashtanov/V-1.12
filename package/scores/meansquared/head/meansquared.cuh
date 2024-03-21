#pragma once

#include "kernel/head/train.cuh"

void meansquared_str_config(Config_t * config, char * key, char * value);
void meansquared_mk(Train_t * train);
void meansquared_free(Train_t * train);

//	==== Loss controle ====
void meansquared_score_th11(Train_t * train);
void meansquared_score(Train_t * train);

void meansquared_loss_th11(Train_t * train);
void meansquared_loss(Train_t * train);

void meansquared_dloss_th11(Train_t * train);
void meansquared_dloss(Train_t * train);

void meansquared_ddloss_th11(Train_t * train);
void meansquared_ddloss(Train_t * train);