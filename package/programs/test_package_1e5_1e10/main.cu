#include "package/package.cuh"

int main(int argc, char ** argv) {
	if (system("python3 -m package.programs.test_package_1e5_1e10.main") != 0) {
		ERR("Sortie avec erreur");
	};
}