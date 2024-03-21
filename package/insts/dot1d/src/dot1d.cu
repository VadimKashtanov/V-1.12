#include "package/insts/dot1d/head/dot1d.cuh"

void dot1d_check(Config_t * inst) {
	//>0 <==> >= 1
	uint * param = inst->param;
	
	if (param[0] == 0) raise(SIGINT);
	if (param[1] == 0) raise(SIGINT);
	if (param[2] >= ACTIVS-1) raise(SIGINT);
};

void dot1d_cpu(Cpu_t * cpu, Config_t * inst, uint time) {
	Mdl_t * mdl = cpu->mdl;

	uint Ax = inst->param[0],			\
		 Yx = inst->param[1],			\
		 activ = inst->param[2],		\
		 istart = inst->param[3],		\
		 ystart = inst->param[4],		\
		 wstart = inst->param[5];

	float * var = cpu->var;
	float * weight = mdl->weight;

	float _tmp;
	uint _inp=time*mdl->total + istart,	\
		 _w=wstart;
	
	for (uint y=0; y < Yx; y++) {
		_tmp = 0;
		//Scalar product of 2 vectors in input (A) and weight (B)
		for (uint i=0; i < Ax; i++) {
			_tmp += var[_inp + i] * weight[_w + i*Yx + y];	//weight[_w + i*Yx];
			//printf("%f * %f\n", var[_inp + i], weight[_w + i*Yx + y]);
		}

		//Adding bias
		_tmp += weight[wstart + Ax*Yx + y];
		
		//Activation
		ACTIVATION_USE(activ, _tmp, _tmp);
		
		//Write it to Y
		var[time*mdl->total + ystart + y] = _tmp;

		//Next colon of weights
		//_w++;
	}
};

//	==========	Utiliser F(x) avec GPU ==========

void dot1d_use(Use_t * use, Config_t * inst, uint time) {
	dot1d_use_call_mode_th11(use, inst, time);
};

//	========== F(x)&DF(x) ==========

void dot1d_forward(Train_t * train, Config_t * inst, uint time, uint start_seed) {
	dot1d_forward_call_mode_th11(train, inst, time, start_seed);
};

void dot1d_backward(Train_t * train, Config_t * inst, uint time, uint start_seed) {
	dot1d_backward_call_mode_th11(train, inst, time, start_seed);
};

//	========== DDF(x) ==============

void dot1d_forward2(Train_t * train, Config_t * inst, uint time, uint start_seed) {
	dot1d_forward2_call_mode_th11(train, inst, time, start_seed);
};

void dot1d_backward2(Train_t * train, Config_t * inst, uint time, uint start_seed) {
	dot1d_backward2_call_mode_th11(train, inst, time, start_seed);
};

void dot1d_backward_of_backward2(Train_t * train, Config_t * inst, uint dw, uint time, uint start_seed) {
	dot1d_backward_of_backward2_call_mode_th11(train, inst, dw, time, start_seed);
};

void dot1d_backward_of_forward2(Train_t * train, Config_t * inst, uint dw, uint time, uint start_seed) {
	dot1d_backward_of_forward2_call_mode_th11(train, inst, dw, time, start_seed);
};