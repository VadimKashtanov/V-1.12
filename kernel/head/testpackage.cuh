#pragma once

#include "train.cuh"
#include "use.cuh"
#include "cpu.cuh"

//#include "analyse_procedurale.cuh"

Data_t * load_test_data(FILE * fp);
float * load_float_array(uint len, FILE * fp);

bool test_package_compare_cpu_and_gpu(float * cpu0, float * gpu_d, uint count);
bool test_package_compare_cpu_and_cpu(float * cpu0, float * cpu1, uint count);

Train_t * test_package_load_train(FILE * file);

void test_inst(FILE * fp);
void test_score(FILE * fp);
void test_opti(FILE * fp);
void test_gtic(FILE * fp);