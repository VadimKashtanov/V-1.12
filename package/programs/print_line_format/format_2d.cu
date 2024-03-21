#include "package/programs/print_line_format/print_line_format.cuh"

static void max_and_min(float * arr, uint len, float * _max, float * _min) {
	float __max = 0;
	float __min = arr[0];

	for (uint i=0; i < len; i++) {
		if (arr[i] > __max)
			__max = arr[i];

		if (arr[i] < __min)
			__min = arr[i];
	}

	*_max = __max;
	*_min = __min;
};

void format_2dsquare(float * start_addres, uint size) {
	if (sqrt(size) != round(sqrt(size))) {
		ERR("Il faut un inputs qui a une racine carre entiere, afin de dessiner un carree")
	}

	uint len = (uint)sqrt(size);

	float _min, _max;

	max_and_min(start_addres, size, &_max, &_min);

	uint val;

	for (uint i=0; i < len; i++) {
		for (uint j=0; j < len; j++) {
			val = (uint)(round((start_addres[i*len + j] - _min)/(_max - _min) * 255.0));
			printf("\033[48;2;%i;%i;%im  \033[0m", val, val, val);
		}
		printf("\n");
	}
};