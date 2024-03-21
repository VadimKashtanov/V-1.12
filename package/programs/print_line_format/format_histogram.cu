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

void format_histogram(float * start_addres, uint size) {
	float _min, _max;

	max_and_min(start_addres, size, &_max, &_min);

	uint val;
	for (uint i=0; i < size; i++) {
		printf("%3.i | ", i);
		val = (uint)(round((start_addres[i] - _min)/(_max - _min) * 255.0));
		for (uint j=0; j < (uint)(round((start_addres[i] - _min)/(_max - _min) * 20.0)); j++)
			printf("\033[48;2;%i;%i;%im  \033[0m", val, val, val);
		for (uint j=(uint)(round((start_addres[i] - _min)/(_max - _min) * 20.0)); j < 20; j++)
			printf("  ");
		printf("  %.3g\n", start_addres[i]);
	}
};