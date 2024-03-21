#include "package/package.cuh"

//	./optimize_mdl config_file

//	Regarder compile.py pour obtenire la maniere de configurer le fichier de config

/*

Algorithme:

Fix <model> <test_data>

for echope in echopes:
	for no_test in no_test_pass:

		batch = random() % batchs
		for same_batch in repeats:
			init()
			forward()
			dloss()
			backward()
			update()
			
	test()

	if max(score) < limit_score:
		break

	if echope % each_n_echopes == 0:
		select()
*/

char * mdl_file = 0;
char * data_file = 0;
//char * test_data_file = 0;
char * out_file = 0;

uint echopes = 1;
uint no_test_passs = 1;
uint same_start_seed_runs = 1;
uint repeats = 1;

uint sets = 1;
uint is_train_random = 1;

float limite_score = 0.0;

uint opti_algo = 0;
char ** opti_args_keys = 0;
char ** opti_args_values = 0;

uint score_algo = 0;
char ** score_args_keys = 0;
char ** score_args_values = 0;

uint echo_weights = 0;
uint echo_vars = 0;
uint echo_locds = 0;
uint echo_grads = 0;
uint echo_meands = 0;

uint read_uint(FILE * fp) {
	uint ret;
	fread(&ret, sizeof(uint), 1, fp);
	return ret;
};

float read_float(FILE * fp) {
	float ret;
	fread(&ret, sizeof(float), 1, fp);
	return ret;
};

char * read_str(FILE * fp) {
	uint len = read_uint(fp);

	char * ret = (char*)malloc(len + 1);
	fread(ret, sizeof(char), len, fp);
	ret[len] = '\0';

	return ret;
};

char ** read_strs(FILE * fp, uint len) {
	char ** ret = (char**)malloc(sizeof(char*) * len);

	uint _str_len;
	for (uint i=0; i < len; i++) {
		_str_len = read_uint(fp);
		ret[i] = (char*)malloc(len+1);
		fread(ret[i], sizeof(char), _str_len, fp);
		ret[i][_str_len] = '\0';
	};

	return ret;
};

void free_strs(char ** strs, uint len) {
	for (uint i=0; i < len; i++)
		free(strs[i]);
	free(strs);
};

int main(int argc, char ** argv) {
	assert(argc == 2);

	//
	//	**** Lancer le code python ****
	//

	char * command = (char*)malloc(100 + strlen(argv[1]));
	sprintf(command, "python3 -m package.programs.optimize_smart.main %s", argv[1]);
	if (system(command) != 0) ERR("Command failled");
	free(command);

	//Some variables to process code
	uint opti_args_len, score_args_len;

	//	File where config is written
	FILE * fp = fopen("package/programs/optimize_smart/tmpt", "rb");

	if (fp == 0)
		ERR("Le python ne c'est pas compilé comme il le faut")

	//
	//	**** Read all configuration parameters ****
	//

	mdl_file = read_str(fp);
	data_file = read_str(fp);
	//test_data_file = read_str(fp);
	out_file = read_str(fp);

	echopes = read_uint(fp);
	no_test_passs = read_uint(fp);
	same_start_seed_runs = read_uint(fp);
	repeats = read_uint(fp);

	sets = read_uint(fp);
	is_train_random = read_uint(fp);

	limite_score = read_float(fp);

	//opti
	opti_algo = read_uint(fp);
	opti_args_len = read_uint(fp);
	opti_args_keys = read_strs(fp, opti_args_len);
	opti_args_values = read_strs(fp, opti_args_len);

	//score
	score_algo = read_uint(fp);
	score_args_len = read_uint(fp);
	score_args_keys = read_strs(fp, score_args_len);
	score_args_values = read_strs(fp, score_args_len);

	echo_weights = read_uint(fp);
	echo_vars = read_uint(fp);
	echo_locds = read_uint(fp);
	echo_grads = read_uint(fp);
	echo_meands = read_uint(fp);

	//
	//	**** Build Train_t and all we need ****
	//
	//=====================================
	
		FILE * mdlfp = fopen(mdl_file, "rb");
		Mdl_t * mdl = mdl_fp_load(mdlfp);
		fclose(mdlfp);
	
	//=====================================
	
		//// Load to Ram and Vram
		Data_t * data = data_open(data_file);
		//Data_t * test_data = data_open(test_data_file);

		/*assert(data->lines == test_data->lines);
		assert(data->inputs == test_data->inputs);
		assert(data->outputs == test_data->outputs);*/

		FILE * data_fp = fopen(data_file, "rb");
		//FILE * test_data_fp = fopen(test_data_file, "rb");

		data_cudamalloc(data);
	
	//=====================================
	
		if (sets == 0)
			ERR("sets can't be = to 0")

		Train_t * train = mk_train(mdl, data, sets);

		if (is_train_random == 1) {
			train_random_weights(train, rand()%10000);
		} else {
			//	Random all not-0th sets
			if (sets > 1) train_random_weights_from_mdl(train, rand()%10000);

			//	The zero'th set have to be the mdl one
			train_inject_weight_from_mdl_to_one_set(train, 0);
		}
	
	//=====================================
	
		Opti_t * opti = opti_mk(train, score_algo, opti_algo);

		for (uint i=0; i < opti_args_len; i++)
			opti_opti_set_one_arg(opti, opti_args_keys[i], opti_args_values[i]);
		for (uint i=0; i < score_args_len; i++)
			opti_score_set_one_arg(opti, score_args_keys[i], score_args_values[i]);

	uint start_seed;
	uint batch_train;

	//uint best_set;
	//float best_score, old_best_score;

	//old_best_score = 100000.0;

	printf("# Starting trainning.\n");

/*#if __NVPROFIL__ == true
	cudaProfilerStart();
#endif*/

	float score[train->sets];
	float _max;
	uint _max_id;

	for (uint lp=0; lp < echopes; lp++) {

		//
		//	Train with data
		//

		//data_cudamalloc(data);
		
		start_seed = rand() % 100000;

		for (uint i=0; i < train->sets; i++)
			score[i] = 0;

		for (uint same_start_seed=0; same_start_seed < same_start_seed_runs; same_start_seed++) {
			//Loop
			batch_train = rand() % data->batchs;

			//	Load a batch
			//StartTimer
			data_load_batch(data, data_fp, batch_train);
			data_cudamemcpy(data);
				
			//	Trainning Part
			for (uint i=0; i < repeats; i++) {
				//	Initialise correctly
				train_null_grad_meand(train);
				train_set_input(train);
			
				//	Forward and Backward
				train_forward(train, start_seed);
				opti_dloss(opti);
				train_backward(train, start_seed);

				//	Optimize
				opti_opti(opti);

				//	Compute Score on this
				//train_null_grad_output(train);
				opti_loss(opti);
				
				for (uint i=0; i < train->sets; i++) {
					//printf("%f\n", opti->set_score[i]);
					//raise(SIGINT);
					score[i] += opti->set_score[i];
					//opti->set_score[i] = 0;
				}

				//==== Prints ====
				if (echo_weights)
					train_print_weights(train);
				if (echo_vars)
					train_print_vars(train);
				if (echo_locds)
					train_print_locds(train);
				if (echo_grads)
					train_print_grads(train);
				if (echo_meands)
					train_print_meands(train);
			}
		}
		_max_id = 0;
		_max = score[0] / (same_start_seed_runs * repeats);
		for (uint i=0; i < train->sets; i++) {
			score[i] /= (same_start_seed_runs * repeats);
			if (score[i] < _max) {
				_max = score[i];
				_max_id = i;
			}
		}

		train_cpy_ws_to_mdl(train, _max_id);

		//data_free_cudamalloc(data);

		//
		//	Test Score with test_data (all batchs)
		//

		/*data_cudamalloc(test_data);
		train->data = test_data;

		find_best_set(test_data, test_data_fp, opti, train, &best_set, &best_score);
		if (best_score < old_best_score) {	//on cherche bien a minimiser Loss(f(x), w)
			old_best_score = best_score;

			//	On copy pour avoire tout le temps le meilleur et sauvgarder ceux qui ont les meilleurs scores
			train_cpy_ws_to_mdl(train, best_set);
		}

		data_free_cudamalloc(test_data);
		train->data = data;*/

		printf("Echope : %i/%i (best score:%f, set=%i)\n", lp+1, echopes, _max, _max_id);
	};

	mdlfp = fopen(out_file, "wb");
	mdl_fp_write(mdl, mdlfp);
	fclose(mdlfp);

	//
	//	**** Free all *****
	//

	//free malloc config params
	free(mdl_file);
	free(data_file);
	//free(test_data_file);
	free(out_file);

	free_strs(opti_args_keys, opti_args_len);
	free_strs(opti_args_values, opti_args_len);
	free_strs(score_args_keys, score_args_len);
	free_strs(score_args_values, score_args_len);

	if (system("rm package/programs/optimize_smart/tmpt") != 0) ERR("Command error");

	fclose(fp);

/*#if __NVPROFIL__ == true
	cudaProfilerStop();
#endif*/

	data_free(data);
	train_free(train);
	mdl_free(mdl);
};