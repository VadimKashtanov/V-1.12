#include "package/package.cuh"

//./print_data data.bin
//./print_data data.bin 7   #batch=7

int main(int argc, char ** argv) {
	if (argc == 2) {
		Data_t * data = data_open(argv[1]);

		for (uint i=0; i < data->batchs; i++) {
			printf("################### BATCH = %i ###################\n", i);
			data_open_batch(data, argv[1], i);
			data_print_batch(data);
		}
		data_free(data);
	} else if (argc == 3) {
		Data_t * data = data_open(argv[1]);
		data_open_batch(data, argv[1], atoi(argv[2]));
		data_print_batch(data);
		data_free(data);
	} else {
		ERR("./print_data data.bin <batch> <ligne>");
	}
}