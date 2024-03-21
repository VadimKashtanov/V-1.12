from package.package import *
from package.test_package import TEST_INSTS, TEST_SCORES, TEST_OPTIS, TEST_GTICS
from kernel.py.test_package import test_package

if __name__ == "__main__":
	bins = test_package(SCORES, OPTIS, GTICS, TEST_INSTS, TEST_SCORES, TEST_OPTIS, TEST_GTICS)

	with open("save.bin", 'wb') as co:
		co.write(bins)