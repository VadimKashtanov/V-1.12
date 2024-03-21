#pragma once

#include "kernel/head/train.cuh"

void neutre_str_config(Config_t * config, char * key, char * value);
void neutre_mk(Train_t * train);
void neutre_free(Train_t * train);

//	==== Gtic & Kernels ====
void neutre_gtic_th11(Train_t * train);

void neutre_gtic(Train_t * train);