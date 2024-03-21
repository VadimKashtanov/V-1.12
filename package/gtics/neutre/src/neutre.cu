#include "package/gtics/neutre/head/neutre.cuh"

static uint controle_set(uint set) {
	if (set == 0) {
		ERR("Gtic:Neutre Sets doit etre non null");
	} else if (set > MAX_SETS) {
		ERR("Gtic:Neutre set = %i doit etre plus petit que la limite %i.", set, MAX_SETS);
	}
	
	return set;
};

void neutre_str_config(Config_t * config, char * key, char * value) {
	if (strcmp(key, "set") == 0) config->param[0] = controle_set(atoi(value));
	else ERR("What is %s (of value %s)", key, value);
};

void neutre_mk(Train_t * train) {
	train->sets = controle_set(train->gtic->param[0]);
};

void neutre_free(Train_t * train) {
	train->gtic->ptr = 0;
};

void neutre_gtic(Train_t * train) {
	neutre_gtic_th11(train);
};