#include "package/package.cuh"

int main(int argc, char ** argv) {
	if (argc == 2) {
		char * command = (char*)malloc(100 + strlen(argv[1]));
		sprintf(command, "python3 -m package.programs.cli_stack_model.main %s", argv[1]);
		if (system(command) != 0) ERR("Command failled");
		free(command);
	} else {
		ERR("Please, write 1 argument : the path to the config file");
	}
};