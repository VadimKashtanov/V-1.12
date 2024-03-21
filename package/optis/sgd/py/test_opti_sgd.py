from package.package import INSTS_DICT
from package.insts.fast_model import Fast_1Layer_FeedForward_Mdl
from kernel.py.test_package import Test_OPTI

from kernel.py.config import Config
from kernel.py.etc import *
from package.defaults_tests import default_test_score, default_test_opti, default_test_gtic

seed(0)

class TEST_OPTI_SGD(Test_OPTI):
	MIN_ECHOPES = 1

	calcule_d = True
	calcule_dd = False

	mdl = Fast_1Layer_FeedForward_Mdl(
		inst:=INSTS_DICT['DOT1D'],
		required:=[Ax:=4,Yx:=3,activ:=0]
	).build_weights()

	lines = 2
	
	#	Opti
	config_opti = Config(0, [float_as_uint(0.1)])

	config_score = default_test_score
	config_gtic = default_test_gtic

class PAPIER_OPTI_SGD:
	PASSER = False

	config_gtic = Config(0, [sets:=1])

	weight = [
		1,-2,
		0,
	]

	meand = [
		[2, 0, -1] #apres l'echope 0
	]

	_dd_weight = [
		None		
	]

	apres_echopes = [
		1-2*0.1, -2-0*0.1,
		0--1*0.1
	]