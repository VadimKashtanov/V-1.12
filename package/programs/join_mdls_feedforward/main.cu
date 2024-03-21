#include "package/package.cuh"

//./join_mdls_feedforward output.bin mdl0.bin mdl1.bin mdl2.bin mdl3.bin
//./join_mdls_feedforward output.bin mdl0.bin

char * join_strs(char * a, char * b) {
	char * ret = (char*)malloc(strlen(a) + 1 + strlen(b)+1);

	sprintf(ret, "%s %s", a, b);

	return ret;
};

int main(int argc, char ** argv) {
	if (argc >= 3) {
		char * sum = (char*)malloc(strlen(argv[2])+1);
		memcpy(sum, argv[2], strlen(argv[2])+1);
		
		char * tmp;

		uint argc_ = (uint)argc;

		for (uint i=3; i < argc_; i++) {
			tmp = join_strs(sum, argv[i]);
			free(sum);
			sum = tmp;
		}

		char * command = (char*)malloc(100 + strlen(argv[1]) + strlen(argv[2]));
		sprintf(command, "python3 -m package.programs.join_mdls_feedforward.main %s %s", argv[1], sum);

		system(command);
	} else {
		ERR("Not enought files : ./join_mdls_feedforward <output:file> <mdl0:file> <mdl1:file> ...");
	}
};