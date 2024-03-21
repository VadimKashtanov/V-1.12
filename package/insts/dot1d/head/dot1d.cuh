#pragma once

#include "kernel/head/testpackage.cuh"

#include "package/insts/activation.cuh"

#include "dot1d_th11.cuh"

//th11:
//	each kernel compute completely one output pixel. [No shared] [No consts] [No texture]

//======================= Mdl_t check ============================

void dot1d_check(Config_t * inst);

//======================= Cpu_t forward ===========================

void dot1d_cpu(Cpu_t * cpu, Config_t * inst, uint time);

//======================= Use_t Forward ===========================

//th11<<<>>>
void dot1d_use_call_mode_th11(Use_t * use, Config_t * inst, uint time);

//call
void dot1d_use(Use_t * use, Config_t * inst, uint time);

//======================== Train_t ===============================

//----------- Calcule dS/d[] avec derivee en Chaine du model ------------------

//INST_capable_df = true;

//th11<<<>>>
void dot1d_forward_call_mode_th11(Train_t * train, Config_t * inst, uint time, uint start_seed);
void dot1d_backward_call_mode_th11(Train_t * train, Config_t * inst, uint time, uint start_seed);

//call
void dot1d_forward(Train_t * train, Config_t * inst, uint time, uint start_seed);
void dot1d_backward(Train_t * train, Config_t * inst, uint time, uint start_seed);

//----------- Calcule d(dS/dw)/d[] avec derivee en Chaine de la derivation en Chaine du model ------------------

//INST_capable_ddf = true;

//th11<<<>>>
void dot1d_forward2_call_mode_th11(Train_t * train, Config_t * inst, uint time, uint start_seed);
void dot1d_backward2_call_mode_th11(Train_t * train, Config_t * inst, uint time, uint start_seed);
void dot1d_backward_of_backward2_call_mode_th11(Train_t * train, Config_t * inst, uint dw, uint time, uint start_seed);
void dot1d_backward_of_forward2_call_mode_th11(Train_t * train, Config_t * inst, uint dw, uint time, uint start_seed);

//call
void dot1d_forward2(Train_t * train, Config_t * inst, uint time, uint start_seed);
void dot1d_backward2(Train_t * train, Config_t * inst, uint time, uint start_seed);
void dot1d_backward_of_backward2(Train_t * train, Config_t * inst, uint dw, uint time, uint start_seed);
void dot1d_backward_of_forward2(Train_t * train, Config_t * inst, uint dw, uint time, uint start_seed);