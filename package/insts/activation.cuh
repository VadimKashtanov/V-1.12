#pragma once

#include "package/package.cuh"

#define ACTIVS 7

#define ACTIVATION_USE(activ, _inp, _out) do {			\
	if (activ == 0) _out = 1 / (1 + exp(-_inp));		\
	else if (activ == 1) _out = tanh(_inp);				\
	else if (activ == 2) _out = exp(-_inp*_inp);		\
	else if (activ == 3) _out = _inp * (_inp > 0);		\
	else if (activ == 4) _out = _inp;					\
	else if (activ == 5) _out = tanh(_inp*0.7)*pi/2;	\
	else if (activ == 6) _out = log(1 + exp(_inp));		\
} while (0);											\

#define ACTIVATION_FORWARD(activ, _inp, _out, _locd) do {		\
	if (activ == 0) {						\
		_out = 1 / (1 + exp(-_inp));		\
		_locd = _out*(1 - _out);			\
	} else if (activ == 1) {				\
		_out = tanh(_inp);					\
		_locd = 1 - _out*_out;				\
	} else if (activ == 2) {				\
		_locd = _inp;						\
		_out = exp(-_inp*_inp);				\
		_locd = -2*_locd*_out;				\
	} else if (activ == 3) {				\
		_out = _inp * (_inp > 0);			\
		_locd = _inp > 0;					\
	} else if (activ == 4) {				\
		_out = _inp;						\
		_locd = 1;							\
	} else if (activ == 5) {				\
		_out = tanh(0.7*_inp)*pi/2;			\
		_locd = (1 - _out*_out)*0.7*pi/2;	\
	} else if (activ == 6) {				\
		_out = log(1 + exp(_inp));			\
		_locd = 1 / (1 + exp(-_inp));		\
	}										\
} while (0);								\

#define ACTIVATION_FORWARD_2(activ, _inp, _out, _locd, _locd2) do {		\
	if (activ == 0) {								\
		_out = 1 / (1 + exp(-_inp));				\
		_locd = _out*(1 - _out);					\
		_locd2 = _locd*(1-2*_out);					\
	} else if (activ == 1) {						\
		_out = tanh(_inp);							\
		_locd = 1 - _out*_out;						\
		_locd2 = -2*_out*_locd;						\
	} else if (activ == 2) {						\
		_locd2 = _inp;								\
		_out = exp(-_inp*_inp);						\
		_locd = -2*_locd2*_out;						\
		_locd2 = -2*_out + 4*_locd2*_locd2*_out;	\
	} else if (activ == 3) {						\
		_out = _inp * (_inp > 0);					\
		_locd = _inp > 0;							\
		_locd2 = 0;									\
	} else if (activ == 4) {						\
		_out = _inp;								\
		_locd = 1;									\
		_locd2 = 0;									\
	} else if (activ == 5) {						\
		_out = tanh(0.7*_inp)*pi/2;					\
		_locd = (1 - _out*_out)*0.7*pi/2;			\
		_locd2 = -2*_out*(1 - _out*_out)*0.49*pi2/4;\
	} else if (activ == 6) {						\
		_out = log(1 + exp(_inp));					\
		_locd = 1 / (1 + exp(-_inp));				\
		_locd2 = _locd*(1-2*_out);					\
	}												\
} while (0);										\


//#define ACTIVATION_USE(activ, _inp, _out) do {if (activ == 0) _out = 1 / (1 + exp(-_inp));else if (activ == 1) _out = tanh(_inp);else if (activ == 2) _out = exp(-_inp*_inp);else if (activ == 3) _out = _inp * (_inp > 0);else if (activ == 4) _out = _inp;else if (activ == 5) _out = tanh(_inp*0.7)*pi/2;} while (0);
//#define ACTIVATION_FORWARD(activ, _inp, _out, _locd) do {if (activ == 0) {_out = 1 / (1 + exp(-_inp));_locd = _out*(1 - _out);} else if (activ == 1) {_out = tanh(_inp);_locd = 1 - _out*_out;} else if (activ == 2) {_locd = _inp;_out = exp(-_inp*_inp);_locd = -2*_locd*_out;} else if (activ == 3) {_out = _inp * (_inp > 0);_locd = _inp > 0;} else if (activ == 4) {_out = _inp;_locd = 1;} else if (activ == 5) {_out = tanh(0.7*_inp)*pi/2;_locd = (1 - _out*_out)*0.7*pi/2;}} while (0);