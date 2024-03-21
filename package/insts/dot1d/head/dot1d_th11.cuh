#pragma once

#include "kernel/head/testpackage.cuh"

//	============================ Use =========================

__global__
void dot1d_use_th11(
	uint Ax, uint Yx, uint activ,
	uint time,
	uint total,
	uint istart, uint ystart, uint wstart,
	float * var, float * weight);

//	============================ Forward & Backward =========================

//  --------- Forward -------- 

__global__
void dot1d_forward_th11(
	uint Ax, uint Yx, uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd,
	uint sets);

//  --------- Backward -------- 

__global__
void dot1d_backward_th11(
	uint Ax, uint Yx, uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint sets);

//	============================ Backward of Forward & Backward =========================

//  --------- Forward2 -------- 

__global__
void dot1d_forward2_th11(
	uint Ax, uint Yx, uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart, uint l2start,
	uint total, uint wsize, uint lsize, uint l2size,
	float * var, float * weight, float * locd, float * locd2,
	uint sets);

//  --------- Backward2 -------- 

__global__
void dot1d_backward2_th11(
	uint Ax, uint Yx, uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint sets);

//  --------- Backaward_Of_Backward2 -------- 

__global__
void dot1d_backward_of_backward2_th11(
	uint Ax, uint Yx, uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	float * _dd_var, float * _dd_weight, float * _dd_locd, float * _dd_grad, float * _dd_meand,
	uint sets);

//  --------- Backaward_Of_Forward2 -------- 

__global__
void dot1d_backward_of_forward2_th11(
	uint Ax, uint Yx,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart, uint l2start,
	uint total, uint wsize, uint lsize, uint l2size,
	float * var, float * weight, float * locd, float * locd2, float * grad, float * meand,
	float * _dd_var, float * _dd_weight, float * _dd_locd, float * _dd_grad, float * _dd_meand,
	uint sets);