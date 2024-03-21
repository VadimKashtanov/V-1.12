#include "package/insts/dot1d/head/dot1d.cuh"

void dot1d_use_call_mode_th11(Use_t * use, Config_t * inst, uint time) {
	Mdl_t * mdl = use->mdl;

	uint Ax = inst->param[0],			\
		 Yx = inst->param[1],			\
		 activ = inst->param[2],		\
		 istart = inst->param[3],		\
		 ystart = inst->param[4],		\
		 wstart = inst->param[5];

	dot1d_use_th11<<<dim3(KERN_DIV(Yx,32)),dim3(32)>>>(
		Ax, Yx,
		activ,
		time,
		mdl->total,
		istart, ystart, wstart,
		use->var_d, use->weight_d);
	CUDA_WAIT_KER()
};

//========================================================
//======================== Train_t =======================
//========================================================

//-------------------------- forward ---------------------

void dot1d_forward_call_mode_th11(Train_t * train, Config_t * inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax = inst->param[0],			\
		 Yx = inst->param[1],			\
		 activ = inst->param[2],		\
		 istart = inst->param[3],		\
		 ystart = inst->param[4],		\
		 wstart = inst->param[5],		\
		 lstart = inst->param[6];

	dot1d_forward_th11<<<dim3(KERN_DIV(Yx,16),train->sets),dim3(16,1)>>>(
		Ax, Yx,
		activ,
		time,
		istart, ystart, wstart, lstart,
		mdl->total, mdl->weights, mdl->locds,
		train->_var, train->_weight, train->_locd,
		train->sets);
	CUDA_WAIT_KER()
};

//-------------------------- backward ---------------------

void dot1d_backward_call_mode_th11(Train_t * train, Config_t * inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax = inst->param[0],			\
		 Yx = inst->param[1],			\
		 activ = inst->param[2],		\
		 istart = inst->param[3],		\
		 ystart = inst->param[4],		\
		 wstart = inst->param[5],		\
		 lstart = inst->param[6];

	dot1d_backward_th11<<<dim3(KERN_DIV(Yx,16),train->sets),dim3(16,1)>>>(
		Ax, Yx,
		activ,
		time,
		istart, ystart, wstart, lstart,
		mdl->total, mdl->weights, mdl->locds,
		train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
		train->sets);
	CUDA_WAIT_KER()
};

//===========================================================================
//=========================== Forward Backward 2 ============================
//===========================================================================

void dot1d_forward2_call_mode_th11(Train_t * train, Config_t * inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax = inst->param[0],			\
		 Yx = inst->param[1],			\
		 activ = inst->param[2],		\
		 istart = inst->param[3],		\
		 ystart = inst->param[4],		\
		 wstart = inst->param[5],		\
		 lstart = inst->param[6],		\
		 l2start = inst->param[7];

	dot1d_forward2_th11<<<dim3(KERN_DIV(Yx,16),train->sets),dim3(16,1)>>>(
		Ax, Yx,
		activ,
		time,
		istart, ystart, wstart, lstart, l2start,
		mdl->total, mdl->weights, mdl->locds, mdl->locd2s,
		train->_var, train->_weight, train->_locd, train->_locd2,
		train->sets);

	CUDA_WAIT_KER()
};

void dot1d_backward2_call_mode_th11(Train_t * train, Config_t * inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax = inst->param[0],			\
		 Yx = inst->param[1],			\
		 activ = inst->param[2],		\
		 istart = inst->param[3],		\
		 ystart = inst->param[4],		\
		 wstart = inst->param[5],		\
		 lstart = inst->param[6];

	dot1d_backward2_th11<<<dim3(KERN_DIV(Yx,16),train->sets),dim3(16,1)>>>(
		Ax, Yx,
		activ,
		time,
		istart, ystart, wstart, lstart,
		mdl->total, mdl->weights, mdl->locds,
		train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
		train->sets);

	CUDA_WAIT_KER()
};

void dot1d_backward_of_backward2_call_mode_th11(Train_t * train, Config_t * inst, uint dw, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax = inst->param[0],			\
		 Yx = inst->param[1],			\
		 activ = inst->param[2],		\
		 istart = inst->param[3],		\
		 ystart = inst->param[4],		\
		 wstart = inst->param[5],		\
		 lstart = inst->param[6];

	float * _dd_w_start = train->_dd_weight + dw*train->sets*mdl->weights;

	dot1d_backward_of_backward2_th11<<<dim3(KERN_DIV(Yx,16),train->sets),dim3(16,1)>>>(
		Ax, Yx,
		activ,
		time,
		istart, ystart, wstart, lstart,
		mdl->total, mdl->weights, mdl->locds,
		train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
		train->_dd_var, _dd_w_start, train->_dd_locd, train->_dd_grad, train->_dd_meand,
		train->sets);

	CUDA_WAIT_KER()
};

void dot1d_backward_of_forward2_call_mode_th11(Train_t * train, Config_t * inst, uint dw, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax = inst->param[0],			\
		 Yx = inst->param[1],			\
		 activ = inst->param[2],		\
		 istart = inst->param[3],		\
		 ystart = inst->param[4],		\
		 wstart = inst->param[5],		\
		 lstart = inst->param[6],		\
		 l2start = inst->param[7];

	float * _dd_w_start = train->_dd_weight + dw*train->sets*mdl->weights;

	dot1d_backward_of_forward2_th11<<<dim3(KERN_DIV(Yx,16),train->sets),dim3(16,1)>>>(
		Ax, Yx,
		activ,
		time,
		istart, ystart, wstart, lstart, l2start,
		mdl->total, mdl->weights, mdl->locds, mdl->locd2s,
		train->_var, train->_weight, train->_locd, train->_locd2, train->_grad, train->_meand,
		train->_dd_var, _dd_w_start, train->_dd_locd, train->_dd_grad, train->_dd_meand,
		train->sets);

	CUDA_WAIT_KER();
};