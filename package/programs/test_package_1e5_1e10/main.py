from package.test_package import TEST_INSTS
from package.package import INSTS, SCORES, OPTIS, GTICS
from kernel.py.etc import *
from kernel.py.data import Data
from kernel.py.train import Train
from kernel.py.use import Use

def comparer(a, b, tolerance=0.001):
	return abs(a-b) < tolerance

def comparer_2_listes_2d(lst0, lst1, sets, ws):
	assert len(lst0) == len(lst1)
	err = 0
	for _set in range(sets):
		print(f"\033[43m------ SET = {_set} -----\033[0m")
		for w in range(ws):
			egalite = comparer(lst0[_set*ws + w], lst1[_set*ws + w])
			couleur = ('\033[42m' if egalite else '\033[101m')
			print(f"\033[92mset={_set}\033[0m|\033[94m{w}\033[0m| {couleur} {float(lst0[_set*ws + w])} --- {float(lst1[_set*ws + w])} \033[0m")
	
			if not egalite:
				err = 1
	print("set|w|     dS --- 1e5")
	if err == 1:
		ERR("Les valeurs sont pas les memes")

def comparer_2_listes_3d(lst0, lst1, dws, sets, ws):
	assert len(lst0) == len(lst1)
	err = 0
	for dw in range(dws):
		print(f"\033[42m =========== DW = {dw} =========== \033[0m")
		for _set in range(sets):
			print(f"\033[43m------ SET = {_set} -----\033[0m")
			for w in range(ws):
				pos = dw*sets*ws + _set*ws + w
				egalite = comparer(lst0[pos], lst1[pos])
				couleur = ('\033[42m' if egalite else '\033[101m')
				print(f"\033[93mdw={dw}\033[0m|\033[92mset={_set}\033[0m|\033[94m{w}\033[0m| {couleur} {float(lst0[pos])} --- {float(lst1[pos])} \033[0m")
		
				if not egalite:
					err = 1
	print("dw|set|w|     dSdS --- 1e10")
	if err == 1:
		ERR("Les valeurs sont pas les memes")

def comparer_3_listes_3d(lst0, lst1, lst2, dws, sets, ws):
	assert len(lst0) == len(lst1)
	err = 0
	for dw in range(dws):
		print(f"\033[42m =========== DW = {dw} =========== \033[0m")
		for _set in range(sets):
			print(f"\033[43m------ SET = {_set} -----\033[0m")
			for w in range(ws):
				pos = dw*sets*ws + _set*ws + w
				egalite = comparer(lst0[pos], lst1[pos])
				couleur = ('\033[42m' if egalite else '\033[101m')
				print(f"\033[93mdw={dw}\033[0m|\033[92mset={_set}\033[0m|\033[94m{w}\033[0m| {couleur} {float(lst0[pos])} --- {float(lst1[pos])} --- {float(lst2[pos])} \033[0m")
		
				if not egalite:
					err = 1
	print("dw|set|w|          dSdS      ---    1e5    ----   1e10")
	if err == 1:
		ERR("Les valeurs sont pas les memes")

def S(train, w0, plus0):
	for s in range(train.sets):
		train._weight[s*train.mdl.weights + w0] += plus0

	train.null_D()
	train.restart()
	train.set_inputs()
	train.forward(0)
	#train.loss()
	train.score()

	for s in range(train.sets):
		train._weight[s*train.mdl.weights + w0] -= plus0

	return [i for i in train.set_score]

if __name__ == "__main__":
	test = TEST_INSTS[0]
	assert test.calcule_d and test.calcule_dd

	mdl = test.mdl
	mdl.check()
	assert mdl.allow_ddf()

	lines = test.lines

	seed(0)

	data = Data(
		batchs=1,
		lines=lines,
		_input=[random() for _ in range(mdl.inputs*lines)],
		output=[random() for _ in range(mdl.outputs*lines)]
	)

	train = Train(
		mdl, data,
		test.config_score, test.config_opti, test.config_gtic,
		SCORES, OPTIS, GTICS,
		test.calcule_d,
		test.calcule_dd,
		dws:=list(range(mdl.weights))
	)
	train.randomize(0)

	print("===============================")
	print("=========== dS & 1e5 ==========")
	print("===============================")
			
	train.calculer_dSdw()
	_meand_dS = [m for m in train._meand]
	train.calculer_dw_1e5()
	_meand_1e5 = [m for m in train._meand]
	comparer_2_listes_2d(_meand_dS, _meand_1e5, train.sets, mdl.weights)

	print("===============================")
	print("========= ddS & 1e10 ==========")
	print("===============================")

	#
	train.calculer_dSdwdw()
	_dd_weight_dSdS = [m for m in train._dd_weight]
	#
	#train.calculer_dwdw_1e5()
	#_dd_weight_1e5 = [m for m in train._dd_weight]
	#
	train.calculer_dwdw_1e10()
	_dd_weight_1e10 = [m for m in train._dd_weight]

	comparer_2_listes_3d(
		_dd_weight_dSdS, _dd_weight_1e10,
		mdl.weights, train.sets, mdl.weights)

	#comparer_3_listes_3d(
	#	_dd_weight_dSdS, _dd_weight_1e5, _dd_weight_1e10,
	#	mdl.weights, train.sets, mdl.weights)