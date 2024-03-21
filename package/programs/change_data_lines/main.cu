#include "package/package.cuh"

//	./change_data_lines data.file lines

/*

Befor change:

	batchs = batchs   =>  batchs = batchs * lines
	lines = lines     =>  lines = 1

Then Check
	
	if the new lines divide `batchs*lines`   (batchs*lines) % new_lines == 0  (congrue a zéro)

Change
	Batchs = batch*lines/new_lines
	Lines = new_lines

*/

int main(int argc, char ** argv) {
	assert(argc == 3);

	uint new_lines = atoi(argv[2]);

	FILE * fp = fopen(argv[1], "rb");

	if (fp == 0)
		ERR("File \"%s\" doesn't exists", argv[1]);

	//
		uint batchs, lines;
		//fread(&batchs, sizeof(uint), 2, fp);
		fread(&batchs, sizeof(uint), 1, fp);
		fread(&lines, sizeof(uint), 1, fp);
	//

	fclose(fp);

	if ((batchs*lines) % new_lines == 0) {
		batchs = lines*batchs/new_lines;
		lines = new_lines;

		//	Replace only batchs and lines
		fp = fopen(argv[1], "r+b");
		//fwrite(&batchs, sizeof(uint), 2, fp);
		fwrite(&batchs, sizeof(uint), 1, fp);
		fwrite(&lines, sizeof(uint), 1, fp);
		fclose(fp);
	} else {
		ERR("Can't batchs*lines %% new_lines == %i   not 0", (batchs*lines) % new_lines);
	}
}