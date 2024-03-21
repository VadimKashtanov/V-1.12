from package.package import INSTS_DICT
from package.insts.fast_model import Fast_1Layer_FeedForward_Mdl
from kernel.py.test_package import Test_INST

from kernel.py.config import Config
from kernel.py.etc import *
from package.defaults_tests import default_test_score, default_test_opti, default_test_gtic

seed(0)

class TEST_INST_DOT1D(Test_INST):
	calcule_d = True
	calcule_dd = True

	mdl = Fast_1Layer_FeedForward_Mdl(
		inst:=INSTS_DICT['dot1d'],
		required:=[Ax:=3,Yx:=2,activ:=1]
	).build_weights()

	### compare .mdl() and .forward()
	###		ex: drop_rate=0
	mdl_mdl_forward_compare = Fast_1Layer_FeedForward_Mdl(
		inst:=INSTS_DICT['dot1d'],
		required:=[Ax:=3,Yx:=2,activ:=4]
	).build_weights()

	lines = 2

	config_score = default_test_score
	config_opti = default_test_opti
	config_gtic = default_test_gtic

#	Donc pour le mdl de test_inst
#	Comme si Lignes = 1 (set = 1 aussi evidement)
class PAPIER_INST_DOT1D:
	PASSER = False
	
	mdl = Fast_1Layer_FeedForward_Mdl(
		inst:=INSTS_DICT['dot1d'],
		required:=[Ax:=3,Yx:=2,activ:=4]
	).build_weights()

	config_gtic = Config(0, [sets:=1])
	
	lignes = 1

	weight = [
		1,-1,
		4,0.5,
		0,1,

		-1,0
	]

	inp = [
		[1,2,3]
	]

	var = [
	#	entree | sortie
		1,2,3,   8,3
	]
