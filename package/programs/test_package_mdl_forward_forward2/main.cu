#include "package/package.cuh"

int main(int argc, char ** argv) {
	if (system("python3 -m package.programs.test_package_mdl_forward_forward2.main") != 0) {
		ERR("Sortie avec erreur");
	};
}