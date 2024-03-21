#include "package/package.cuh"

int main(int argc, char ** argv) {
	if (system("python3 -m package.programs.test_package_papier.main") != 0) ERR("./test_package_papier finis en erreure")
}