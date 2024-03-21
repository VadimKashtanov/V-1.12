#include "package/package.cuh"

int main(int argc, char ** argv) {
	if (system("python3 -m package.programs.test_package_mdl_1e5.main") != 0) {
		ERR("Sortie avec erreur");
	};
}