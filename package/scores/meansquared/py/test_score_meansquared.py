from package.package import INSTS_DICT
from package.insts.fast_model import Fast_1Layer_FeedForward_Mdl
from kernel.py.test_package import Test_SCORE

from kernel.py.config import Config
from kernel.py.etc import *
from package.defaults_tests import default_test_score, default_test_opti, default_test_gtic

seed(0)

class TEST_SCORE_MEANSQUARED(Test_SCORE):
	calcule_d = True
	calcule_dd = True

	mdl = Fast_1Layer_FeedForward_Mdl(
		inst:=INSTS_DICT['DOT1D'],
		required:=[Ax:=4,Yx:=3,activ:=0]
	).build_weights()

	lines = 2

	#	Score
	config_score = Config(0, [])

	config_opti = default_test_opti
	config_gtic = default_test_gtic

class PAPIER_SCORE_MEANSQUARED:
	PASSER = False

	config_gtic = Config(0, [sets:=1])

	lignes = 1

	get = [
		4, 5,
	]
	want = [
		1, -3,
	]

	#======== --> =========

	_grad_loss = [
		(4-1)**2/2, (5--3)**2/2,
	]

	score = [
		( (4-1)**2/2 + (5--3)**2/2)
	][0] #car 1 seul set (pour l'instant j'ai pas implément avec plusieurs)

	#======== <-- =========

	_grad_dloss = [
		(4-1), (5--3)
	]

	#======== --> =======

	_dd_grad_ddloss = [
		1, 2
	]

	_dd_var_ddloss = [
		1, 2
	]