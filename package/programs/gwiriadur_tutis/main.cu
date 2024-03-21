#include "package/package.cuh"

int main(int argc, char ** argv) {
	const char* programmes[7] = {
	//	Tester que les f(x) en Python correspondent a ce qui est demandé sur papier
		"./test_package_papier",
		"./test_package_mdl_forward_forward2",
	//	Tester que les scores dl et ddl soient coherents via 1e5 et 1e10
		"./test_package_score_1e5",
		"./test_package_score_1e10",
	//	Tester que df(x) et ddf(x) de Python correspondent a f(x+1e-5) et df(x+1e-5)
		"./test_package_mdl_1e5",
		"./test_package_mdl_1e10",
	//	Test de Train_t en Python
		"./test_package_1e5_1e10",
	//	Tester que C/Cuda implémente bien le code python
		"./test_package",
	};

	for (uint i=0; i < 7; i++) {
		printf("###########################################################################\n");
		printf("###########################################################################\n");
		printf("################################ \033[1;90,42m%s\033[0m ########################################\n", programmes[i]);
		printf("###########################################################################\n");
		printf("###########################################################################\n");
		if (system(programmes[i]) != 0) {
			ERR("En Erreur %s le programme c'est terminé", programmes[i]);
		}
	}
	return 0;
}