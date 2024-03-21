#include "kernel/head/etc.cuh"

FILE * SAFE_FOPEN(const char * file, const char * mode) {
	FILE * fp = fopen(file, mode);
	if (fp == 0)
		ERR("Le fichier %s n'existe pas.", file);
	return fp;
};

uint read_uint(FILE * fp) {
	uint _123;
	fread(&_123, sizeof(uint), 1, fp);
	return _123;
};

uint read_123(FILE * fp) {
	uint _123;
	fread(&_123, sizeof(uint), 1, fp);
	return (uint)(_123 == 123);
};

void write_123(FILE * fp) {
	uint _123 = 123;
	fwrite(&_123, sizeof(uint), 1, fp);
};

//#define is_123(fp) do {if (read_uint(fp) != 123) ERR("Etait attendu 123");}while(0);
void is_123(FILE * fp) {
	uint _123 = read_uint(fp);

	if (_123 != 123) {
		float c;
		memcpy(&c, &_123, sizeof(float));
		ERR("Etait attendu 123. Obtenu : (uint)%i, (float)%f", _123, (float)c);
	}
}

/*void etc_parse_arguments(uint argc, char ** argv, uint paramc, const char ** paramv, char ** correspondance) {
	for (uint i=0; i < argc; i++) {
		//	Find the correspondance
		for (uint j=0; j < paramc; j++) {
			if (strcmp(argv[i+1], paramv[j]) == 0) {	//'-sets 3' to skip the '-'
				correspondance[j] = argv[i];
				i++;
				break;
			}
		}

	}
};*/

static void terminale_plateau_pixels(char ** premier_text, bool ** pixs, char ** dig0s, char ** dig1s, uint len, uint H) {
	//
	uint _max = strlen(premier_text[0]);
	for (uint i=0; i < H; i++)
		if (strlen(premier_text[i]) > _max) _max = strlen(premier_text[i]);
	
	//
	for (uint y=0; y < H; y++) {
		printf("%s", premier_text[y]);
		for (uint j=0; j < (1+_max-strlen(premier_text[y])); j++) printf(" ");
		//
		printf("\033[100;94;4m ");
		for (uint i=0; i < len; i++)
			printf("\033[%i;94;4m%c%c\033[100;4m ", (pixs[y][i]== 0 ? 100 : 107), dig0s[y][i], dig1s[y][i]);
		printf("\033[0m\n");
	}
};

void term_plot(float * values, uint len, uint H) {
	if (len > 100 || len == 0)
		ERR("La taille d'un array doit etre entre 1 et 100. Ici elle est de %i", len);

	if (H <= 1)
		ERR("H doit etre > 1");

	uint h = H - 1;
	printf("Plotting [%.5g", values[0]);
	for (uint i=1; i < len; i++) printf(", %.5g", values[i]);
	printf("]\n");

	//	Valeur max et min
	float _min = values[0];
	float _max = values[0];
	for (uint i=1; i < len; i++) {
		if (values[i] > _max) _max = values[i];
		if (values[i] < _min) _min = values[i];
	}

	//	Maintenant valeur entre 0.0 et 1.0
	float norm_values[len];
	for (uint i=0; i < len; i++)
		norm_values[i] = (values[i] - _min)/(_max - _min);

	//	Construction des digits et texts premier
	char ** premier_text = (char**)malloc(sizeof(char*) * H);
	bool ** pixs = (bool**)malloc(sizeof(bool*) * H);
	char ** dig0s = (char**)malloc(sizeof(char*) * H);
	char ** dig1s = (char**)malloc(sizeof(char*) * H);

	for (uint y=0; y < H; y++) {
		pixs[y] = (bool*)malloc(sizeof(bool) * len);
		dig0s[y] = (char*)malloc(sizeof(char) * len);
		dig1s[y] = (char*)malloc(sizeof(char) * len);
		for (uint x=0; x < len; x++) {
			dig0s[y][x] = ' ';
			dig1s[y][x] = ' ';
			pixs[y][x] = 0;
		}
	}

	uint hauteur;
	for (uint i=0; i < len; i++) {
		hauteur = (uint)(H - (norm_values[i]*h) - 1);

		dig0s[hauteur][i] = 48 + (uint)roundf((i - (i%10))/10);
		dig1s[hauteur][i] = 48 + i%10;
		for (uint k=hauteur; k < H; k++) {
			pixs[k][i] = 1;
		}
	};

	uint _strlen;
	float val;
	for (uint y=0; y < H; y++) {
		val = (_max - (_max-_min)*y/h);
		_strlen = snprintf(NULL, 0, "%.5g", val);
		premier_text[y] = (char*)malloc(_strlen + 1);
		snprintf(premier_text[y], _strlen + 1, "%f", val);
	};

	terminale_plateau_pixels(premier_text, pixs, dig0s, dig1s, len, H);

	for (uint i=0; i < H; i++)
		free(premier_text[i]);
	free(premier_text);

	for (uint y=0; y < H; y++) {
		free(pixs[y]);
		free(dig0s[y]);
		free(dig1s[y]);
	}
	free(pixs);
	free(dig0s);
	free(dig1s);
};

//=============================================================================================

float cuda_get_float(float * arr_d, uint pos) {
	float res;
	SAFE_CUDA(cudaMemcpy(&res, arr_d + pos, sizeof(float) * 1, cudaMemcpyDeviceToHost));
	return res;
};

uint cuda_get_uint(uint * arr_d, uint pos) {
	uint res;
	SAFE_CUDA(cudaMemcpy(&res, arr_d + pos, sizeof(uint) * 1, cudaMemcpyDeviceToHost));
	return res;
};