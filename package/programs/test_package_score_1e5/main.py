from package.test_package import TEST_SCORES, PAPIER_SCORES
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
		print(f"\033[43m------ Ligne = {_set} -----\033[0m")
		for w in range(ws):
			egalite = comparer(lst0[_set*ws + w], lst1[_set*ws + w])
			couleur = ('\033[42m' if egalite else '\033[101m')
			print(f"\033[92ml={_set}\033[0m|\033[94m{w}\033[0m| {couleur} {float(lst0[_set*ws + w])} --- {float(lst1[_set*ws + w])} \033[0m")
	
			if not egalite:
				err = 1
	print("l|o|     dL --- 1e5")
	if err == 1:
		ERR("Les valeurs sont pas les memes")

def S(train, w0, plus0):
	ostart = train.mdl.total - train.mdl.outputs
	for l in range(train.data.lines):
		train._var[l*train.mdl.total + ostart + w0] += plus0

	train.loss()
	train.score()

	for l in range(train.data.lines):
		train._var[l*train.mdl.total + ostart + w0] -= plus0

	return [train._grad[l*train.mdl.total + ostart + i] for l in range(train.data.lines) for i in range(train.mdl.outputs)]

if __name__ == "__main__":
	for test, papier in zip(TEST_SCORES, PAPIER_SCORES):
		print("\033[46m ====================================================================== \033[0m")
		print(f"\033[46m ========================\033[94m {test.__name__} \033[0m\033[46m============================= \033[0m")
		print("\033[46m ====================================================================== \033[0m")
		if test.calcule_d and not papier.PASSER:
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
				test.config_score, test.config_opti, papier.config_gtic,
				SCORES, OPTIS, GTICS,
				test.calcule_d,
				test.calcule_dd,
				[]
			)
			train.randomize(0)
			assert train.sets == 1

			train.null_for_dS()
			train.set_inputs()
			train.forward(0)
			train.dloss()

			ostart = train.mdl.total - train.mdl.outputs
			
			_dloss_val = [train._grad[l*train.mdl.total + ostart + i] for l in range(train.data.lines) for i in range(train.mdl.outputs)]

			train.null_for_dS()
			train.set_inputs()
			train.forward(0)

			_dloss_1e5 = [0 for _ in _dloss_val]

			for o in range(mdl.outputs):
				loss1 = S(train, o, 1e-5)
				loss0 = S(train, o, 0)

				for l in range(train.data.lines):
					wpos = l*mdl.outputs + o
					_dloss_1e5[wpos] = (loss1[wpos] - loss0[wpos])*1e5

				del loss1, loss0

			comparer_2_listes_2d(_dloss_val, _dloss_1e5, train.data.lines, mdl.outputs)

		else:
			print("\033[43m ========================================= \033[0m")
			print("\033[43m ========= Pas de ddF implémenté ========= \033[0m")
			print("\033[43m ========================================= \033[0m")