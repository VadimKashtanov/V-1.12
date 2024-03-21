#include "package/package.cuh"

//./random_data file.bin 5 3 1 7
//	batchs = 5
//	lines = 3
//	inputs = 1
//	outputs = 7

//from 0.0 to 1.0

int main(int argc, char ** argv)
{
	if (argc == 6) {
		FILE * fp = fopen(argv[1], "wb");

		if (fp == 0) {
			ERR("File %s doesn't exists", argv[1]);
		}

		uint batchs = atoi(argv[2]);
		uint lines = atoi(argv[3]);
		uint inputs = atoi(argv[4]);
		uint outputs = atoi(argv[5]);

		fwrite(&batchs, sizeof(uint), 1, fp);
		fwrite(&lines, sizeof(uint), 1, fp);
		fwrite(&inputs, sizeof(uint), 1, fp);
		fwrite(&outputs, sizeof(uint), 1, fp);

		srand(0);

		float tmpt;
		for (uint i=0; i < batchs*lines*(inputs + outputs); i++) {
			//Il y a pas d'ordre, je dois juste ecrire un certain nombre de random foats
			
			tmpt = ((float)(rand() % 1000))/1000;

			fwrite(&tmpt, sizeof(float), 1, fp);
		}

		fclose(fp);

	} else {
		ERR("Not 5 arguments. <data file:str> <batchs:int> <lines:int> <inputs:int> <outputs:int>");
	}
};