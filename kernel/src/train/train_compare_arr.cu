#include "kernel/head/train.cuh"

static uint print_arr_dim2(
	Separators_t * sep, float * arr, float * arr1,
	const char * dim0_str, //char * dim1_str,
	uint dim0, uint dim1,
	float tolerance)
{
	int lbl;
	uint pos;

	uint erreurs = 0;

	for (uint d0=0; d0 < dim0; d0++) {
		printf("\033[%im||\033[0m", (d0 % 2 ? 92 : 91));	// '||' de la ligne
		printf("%s #%i ============= \n", dim0_str, d0);
		for (uint d1=0; d1 < dim1; d1++) {
			lbl = find_sep(sep, d1);

			if (lbl != -1) {
				printf("\033[%im||\033[0m", (d0 % 2 ? 92 : 91));	// '||' de la ligne
				printf("|| (%i) %s\n", d1, sep->labels[lbl]);
			}

			pos = d0*dim1 + d1;
			
			printf("\033[%im||\033[0m", (d0 % 2 ? 92 : 91));	// '||' de la ligne
			
			if (compare_floats(arr[pos], arr1[pos], tolerance)) {
				printf("|| %i |  \033[42m %f --- %f \033[0m \n", d1, arr[pos], arr1[pos]);
			} else {
				printf("|| %i |  \033[41m %f --- %f \033[0m \n", d1, arr[pos], arr1[pos]);
				erreurs = 1;
			}
		}
	}

	printf("        c/cuda ---- python\n");

	return erreurs;
};

static uint print_arr_dim3(
	Separators_t * sep, float * arr, float * arr1,
	const char * dim0_str, const char * dim1_str, //char * dim2_str,
	uint dim0, uint dim1, uint dim2,
	float tolerance)
{
	int lbl;
	uint pos;

	uint erreurs = 0;

	for (uint d0=0; d0 < dim0; d0++) {
		printf("\033[%im||\033[0m", (d0 % 2 ? 92 : 91));
		printf("%s = %i ################### \n", dim0_str, d0);
		for (uint d1=0; d1 < dim1; d1++) {
			printf("\033[%im||\033[0m", (d0 % 2 ? 92 : 91));	// '||' de la ligne
			printf("\033[%im||\033[0m", (d1 % 2 ? 93 : 96)); // '||' du set
			printf("%s #%i ============= \n", dim1_str, d1);
			for (uint d2=0; d2 < dim2; d2++) {
				lbl = find_sep(sep, d2);

				if (lbl != -1) {
					printf("\033[%im||\033[0m", (d0 % 2 ? 92 : 91));	// '||' de la ligne
					printf("\033[%im||\033[0m", (d1 % 2 ? 93 : 96)); // '||' du set
					printf("|| (%i) %s\n", d2, sep->labels[lbl]);
				}

				pos = d0*dim1*dim2 + d1*dim2 + d2;
				
				printf("\033[%im||\033[0m", (d0 % 2 ? 92 : 91));	// '||' de la ligne
				printf("\033[%im||\033[0m", (d1 % 2 ? 93 : 96)); // '||' du set
				
				if (compare_floats(arr[pos], arr1[pos], tolerance)) {
					printf("|| %i |  \033[42m %f --- %f \033[0m \n", d2, arr[pos], arr1[pos]);
				} else {
					printf("|| %i |  \033[41m %f --- %f \033[0m \n", d2, arr[pos], arr1[pos]);
					erreurs = 1;
				}
			}
		}
	}

	printf("         c/cuda ---- python\n");

	return erreurs;
};

static const char * arrs[11] = {
	"_weight", "_var", "_locd", "_grad", "_meand", "_dd_weight", "_dd_var", "_dd_locd", "_dd_grad", "_dd_meand", "_locd2"
};

static int find(const char * nom) {
	for (uint i=0; i < 11; i++)
		if (strcmp(nom, arrs[i]) == 0)
			return i;
	return -1;
};

uint train_eq_arr(Train_t * train, const char * nom, float * cpu_arr, float tolerance) {
	Mdl_t * mdl = train->mdl;

	uint sets = train->sets;
	uint lines = train->lines;
	uint total = mdl->total;
	uint wsize = mdl->weights;
	uint lsize = mdl->locds;
	uint l2size = mdl->locd2s;

	uint dws = train->dws;

	uint arrs_tens[11][3] = {
		{sets, wsize, 0},
		{lines, sets, total},
		{lines, sets, lsize},

		{lines, sets, total},
		{sets, wsize, 0},

		{dws, sets, wsize},
		{lines, sets, total},
		{lines, sets, lsize},
		{lines, sets, total},
		{sets, wsize, 0},
		{lines, sets, l2size}
	};

	float * arrs_ptrs[11] = {
		train->_weight, train->_var, train->_locd,
		train->_grad, train->_meand,
		train->_dd_weight, train->_dd_var, train->_dd_locd, train->_dd_grad, train->_dd_meand, train->_locd2
	};

	int pos = find(nom);

	if (pos == -1) ERR("train->%s n'existe pas", nom);
	if (arrs_ptrs[pos] == 0) ERR("train->%s == 0", nom);

	uint len = arrs_tens[pos][0] * arrs_tens[pos][1] * (arrs_tens[pos][2] == 0 ? 1 : arrs_tens[pos][2]);

	float * tmp_cpu = (float*)malloc(sizeof(float) * len);
	SAFE_CUDA(cudaMemcpy(tmp_cpu, arrs_ptrs[pos], sizeof(float) * len, cudaMemcpyDeviceToHost));

	for (uint i=0; i < len; i++)
		if (compare_floats(tmp_cpu[i], cpu_arr[i], tolerance) != 1)
			return 0;

	free(tmp_cpu);

	return 1;
};

void train_print_compare_arr(Train_t * train, const char * nom, float * cpu_arr, float tolerance) {
	Mdl_t * mdl = train->mdl;

	uint sets = train->sets;
	uint lines = train->lines;
	uint total = mdl->total;
	uint wsize = mdl->weights;
	uint lsize = mdl->locds;
	uint l2size = mdl->locd2s;

	uint dws = train->dws;

	Separators_t * wsep = mdl->wsep, * vsep = mdl->vsep, * lsep = mdl->lsep, * l2sep = mdl->l2sep;

	uint arrs_dim[11] = {
		2, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2
	};

	uint arrs_tens[11][3] = {
		{sets, wsize, 0},
		{lines, sets, total},
		{lines, sets, lsize},

		{lines, sets, total},
		{sets, wsize, 0},

		{dws, sets, wsize},
		{lines, sets, total},
		{lines, sets, lsize},
		{lines, sets, total},
		{sets, wsize, 0},
		{lines, sets, l2size}
	};

	const char * arr_tens_lbl[11][2] = {
		{"Set", 0},
		{"Line", "Set"},
		{"Line", "Set"},

		{"Line", "Set"},
		{"Set", 0},

		{"DW", "Set"},
		{"Line", "Set"},
		{"Line", "Set"},
		{"Line", "Set"},
		{"Set", 0},
		{"Line", "Set"}
	};

	float * arrs_ptrs[11] = {
		train->_weight, train->_var, train->_locd,
		train->_grad, train->_meand,
		train->_dd_weight, train->_dd_var, train->_dd_locd, train->_dd_grad, train->_dd_meand, train->_locd2
	};

	Separators_t * arrs_sep[11] = {
		wsep, vsep, lsep,
		vsep, wsep,
		wsep, vsep, lsep, vsep, wsep, l2sep
	};

	int pos = find(nom);

	if (pos == -1) ERR("train->%s n'existe pas", nom);
	if (arrs_ptrs[pos] == 0) ERR("train->%s == 0", nom);

	uint len = arrs_tens[pos][0] * arrs_tens[pos][1] * (arrs_tens[pos][2] == 0 ? 1 : arrs_tens[pos][2]);

	float * train_cpu = (float*)malloc(sizeof(float) * len);
	SAFE_CUDA(cudaMemcpy(train_cpu, arrs_ptrs[pos], sizeof(float) * len, cudaMemcpyDeviceToHost));

	if (arrs_dim[pos] == 2) {
		print_arr_dim2(arrs_sep[pos], train_cpu, cpu_arr,
			arr_tens_lbl[pos][0],
			arrs_tens[pos][0], arrs_tens[pos][1],
			tolerance
		);
	} else {
		print_arr_dim3(arrs_sep[pos], train_cpu, cpu_arr,
			arr_tens_lbl[pos][0], arr_tens_lbl[pos][1],
			arrs_tens[pos][0], arrs_tens[pos][1], arrs_tens[pos][2],
			tolerance
		);
	}
	free(train_cpu);
};