from package.test_package import TEST_INSTS
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
			print(f"\033[92mset={_set}\033[0m|\033[94m{w}\033[0m| {couleur} {float(lst0[_set*ws + w])} --- {float(lst1[_set*ws + w])} \033[0m")
	
			if not egalite:
				err = 1
	print("set|w|     dS --- 1e5")
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
	for test in TEST_INSTS:
		print("\033[46m ====================================================================== \033[0m")
		print(f"\033[46m ========================\033[94m {test.__name__} \033[0m\033[46m============================= \033[0m")
		print("\033[46m ====================================================================== \033[0m")
		if test.calcule_d:
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
				[]
			)
			train.randomize(0)
			
			train.calculer_dSdw()
			_dw = [i for i in train._meand]

			_meand = [0 for i in train._meand]

			train.ddS_first_null()

			for w in range(mdl.weights):
				scores1 = S(train, w, 1e-5)
				scores0 = S(train, w, 0)

				for _set in range(train.sets):
					wpos = _set*mdl.weights + w
					_meand[wpos] = (scores1[_set] - scores0[_set])/1e-5

				del scores1, scores0

			comparer_2_listes_2d(_dw, _meand, train.sets, mdl.weights)

		else:
			print("\033[43m ========================================= \033[0m")
			print("\033[43m ========= Pas de dF implémenté ========= \033[0m")
			print("\033[43m ========================================= \033[0m")