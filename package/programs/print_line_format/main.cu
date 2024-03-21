#include "package/package.cuh"

#include "package/programs/print_line_format/print_line_format.cuh"

//Formats:
//	2dsquare		Print image. Normalizing from 0 to 255.
//	histogram		Print an histogram, normalizing from 0 to 255.

//Use command
//	./print_line_format data.bin batch line <input format> <output format>
//ex : ./print_line_format data.bin 0 1 2dsquare histogram
//

int main(int argc, char ** argv)
{
	if (argc == 2) {
		if (strcmp(argv[1], "help") == 0) {
			printf("Use : ./print_line_format data.bin batch line <input format> <output format>\n");
			printf("Formats : 2dsquare, histogram\n");
		}
	}

	if (argc != 6)
		ERR("FORMAT : ./print_line_format data.bin batch line <input format> <output format>")

	Data_t * data = data_open(argv[1]);

	FILE * fp = fopen(argv[1], "rb");
	data_load_batch(data, fp, atoi(argv[2]));
	fclose(fp);

	uint batch = atoi(argv[2]);
	uint line = atoi(argv[3]);

	printf("====== Input batch=%i/%i line=%i/%i ======\n", batch, data->batchs,  line, data->lines);
	if (strcmp(argv[4], "2dsquare") == 0) {
		format_2dsquare(data->input + line*data->inputs, data->inputs);
	} else if (strcmp(argv[4], "histogram") == 0) {
		format_histogram(data->input + line*data->inputs, data->inputs);
	} else {
		ERR("What is '%s' format ?\n", argv[4]);
	};

	printf("====== Output batch=%i/%i line=%i/%i ======\n", batch, data->batchs,  line, data->lines);
	if (strcmp(argv[5], "2dsquare") == 0) {
		format_2dsquare(data->output + line*data->outputs, data->outputs);
	} else if (strcmp(argv[5], "histogram") == 0) {
		format_histogram(data->output + line*data->outputs, data->outputs);
	} else {
		ERR("What is '%s' format ?\n", argv[5]);
	};

	data_free(data);
}