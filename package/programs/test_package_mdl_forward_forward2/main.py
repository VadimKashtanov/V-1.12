from package.test_package import TEST_INSTS, PAPIER_INSTS
from package.package import INSTS, SCORES, OPTIS, GTICS
from kernel.py.etc import *
from kernel.py.data import Data
from kernel.py.train import Train
from kernel.py.use import Use

def comparer(a, b, tolerance=0.00001):
	return abs(a-b) < tolerance

def comparer_2_listes_2d(lst0, lst1, sets, ws):
	assert len(lst0) == len(lst1)
	err = 0
	for _set in range(sets):
		print(f"\033[43m------ SET = {_set} -----\033[0m")
		for w in range(ws):
			egalite = comparer(lst0[_set*ws + w], lst1[_set*ws + w])
			couleur = ('\033[42m' if egalite else '\033[101m')
			print(f"set=0|\033[92ml={_set}\033[0m|\033[94m{w}\033[0m| {couleur} {float(lst0[_set*ws + w])} --- {float(lst1[_set*ws + w])} \033[0m")
	
			if not egalite:
				err = 1
	print("s=0|l|i|  train.forward/2 --- mdl (papier)")
	if err == 1:
		ERR("Les valeurs sont pas les memes")

def S(train, w0, plus0):
	for s in range(train.sets):
		train._weight[s*train.mdl.weights + w0] += plus0

	train.null_for_dS()
	train.set_inputs()
	train.forward(0)
	#train.loss()
	train.score()

	for s in range(train.sets):
		train._weight[s*train.mdl.weights + w0] -= plus0

	return [i for i in train.set_score]

if __name__ == "__main__":
	for test, papier in zip(TEST_INSTS, PAPIER_INSTS):
		print("\033[46m ====================================================================== \033[0m")
		print(f"\033[46m ========================\033[94m {test.__name__} \033[0m\033[46m============================= \033[0m")
		print("\033[46m ====================================================================== \033[0m")
		
		mdl = test.mdl_mdl_forward_compare
		mdl.check()
		assert mdl.allow_ddf()

		lines = papier.lignes

		seed(0)
	
		data = Data(
			batchs=1,
			lines=lines,
			_input=[i for inp in papier.inp for i in inp],
			output=[0 for _ in range(mdl.outputs*lines)]
		)

		train = Train(
			mdl, data,
			test.config_score, test.config_opti, papier.config_gtic,
			SCORES, OPTIS, GTICS,
			test.calcule_d,
			test.calcule_dd,
			[]
		)
		if train.sets != 1:
			ERR("Il faut que sets = 1 (voire le gtic)")
		train._weight = papier.weight
			
		train.set_inputs()
	
		train.forward(0)
		print("=================== Forward 1 & papier ================")
		comparer_2_listes_2d(train._var, papier.var, lines, mdl.total)

		train.forward2(0)
		print("=================== Forward 2 & papier ================")
		comparer_2_listes_2d(train._var, papier.var, lines, mdl.total)