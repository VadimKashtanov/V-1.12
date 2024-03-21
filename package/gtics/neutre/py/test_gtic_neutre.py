from package.package import INSTS_DICT
from package.insts.fast_model import Fast_1Layer_FeedForward_Mdl
from kernel.py.test_package import Test_GTIC

from kernel.py.config import Config
from kernel.py.etc import *
from package.defaults_tests import default_test_score, default_test_opti, default_test_gtic

seed(0)

class TEST_GTIC_NEUTRE(Test_GTIC):
	MIN_ECHOPES = 1

	calcule_d = False
	calcule_dd = False

	mdl = Fast_1Layer_FeedForward_Mdl(
		inst:=INSTS_DICT['dot1d'],
		required:=[Ax:=4,Yx:=3,activ:=0]
	).build_weights()

	lines = 2

	config_score = default_test_score
	config_opti = default_test_opti

	#	Gtic
	config_gtic = Config(0, [sets:=3])

class PAPIER_GTIC_NEUTRE:
	'''mdl = Fast_1Layer_FeedForward_Mdl(
		inst:=INSTS_DICT['DOT1D'],
		required:=[Ax:=2,Yx:=1,activ:=0]
	).build_weights()

	weight = [
		1,-2,
		0,
	]

	meand = [
		[2, 0, -1] #apres l'echope 0
	]

	apres_echopes = [
		2-2*0.1, -2-0*0.1,
		0--1*0.1
	]'''