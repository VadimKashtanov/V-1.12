#include "kernel/head/config.cuh"

Config_t * config_mk(uint id, uint params, const uint * param) {
	Config_t * ret = (Config_t*)malloc(sizeof(Config_t));

	ret->id = id;
	ret->params = params;

	ret->param = (uint*)malloc(sizeof(uint) * params);
	memcpy(ret->param, param, sizeof(uint) * params);

	ret->ptr = 0;

	return ret;
};

Config_t * config_load(FILE * fp) {
	is_123(fp);

	uint id, params;
	fread(&id, sizeof(uint), 1, fp);
	fread(&params, sizeof(uint), 1, fp);

	Config_t * ret = (Config_t*)malloc(sizeof(Config_t));

	ret->id = id;
	ret->params = params;

	ret->param = (uint*)malloc(sizeof(uint) * params);
	fread(ret->param, sizeof(uint), params, fp);

	ret->ptr = 0;

	is_123(fp);

	return ret;
};

Config_t * config_open(const char * file) {
	FILE * fp = SAFE_FOPEN(file, "rb");
	Config_t * ret = config_load(fp);
	fclose(fp);
	return ret;
};

void config_write(Config_t * config, FILE * fp) {
	write_123(fp);
	fwrite(&config->id, sizeof(uint), 1, fp);
	fwrite(&config->params, sizeof(uint), 1, fp);
	fwrite(config->param, sizeof(uint), config->params, fp);
	write_123(fp);
};

Config_t * cpy_config(const Config_t * conf) {
	return config_mk(conf->id, conf->params, conf->param);
};

void config_print(Config_t * config) {
	printf("Config_t->id = %i\n", config->id);
	for (uint i=0; i < config->params; i++)
		printf("%i, ", config->param[i]);
	printf("\n");
};

void config_free(Config_t * config) {
	if (config->ptr != 0) ERR("Config_t->ptr != 0. Donc la structure n'a pas ete libere")
	free(config->param);
	free(config);
};