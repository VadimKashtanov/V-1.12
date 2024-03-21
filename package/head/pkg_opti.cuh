#pragma once

#include "kernel/head/testpackage.cuh"

void config_score(Config_t * score, char * key, char * value);
void config_opti(Config_t * opti, char * key, char * value);
void config_gtic(Config_t * gtic, char * key, char * value);