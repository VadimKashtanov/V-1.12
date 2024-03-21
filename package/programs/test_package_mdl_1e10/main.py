from package.test_package import TEST_INSTS
from package.package import INSTS, SCORES, OPTIS, GTICS
from kernel.py.etc import *
from kernel.py.data import Data
from kernel.py.train import Train
from kernel.py.use import Use

'''

df/dx = (f(x+1e-5) - f(x))/1e-5
d(df/dx)/dy = ((f(x+1e-5,y+1e-5) - f(x,y+1e-5))/1e-5 - (f(x+1e-5) - f(x))/1e-5)/1e-5
			= (f(x+,y+) - f(y+) - f(x+) + f()) / 1e-10
			= (df/dx(y+1e-5) - df/dx(y)) / 1e-5
			
'''

def comparer(a, b, tolerance=0.00001):
	return abs(a-b) < tolerance

def comparer_2_listes_3d(lst0, lst1, dws, sets, ws):
	assert len(lst0) == len(lst1)
	err = 0
	for dw in range(dws):
		print(f"\033[46m-------------------- DW = {dw} ----------------------\033[0m")
		for _set in range(sets):
			print(f"\033[43m-------- SET = {_set} -------\033[0m")
			for w in range(ws):
				f0, f1 = lst0[dw*sets*ws + _set*ws + w], lst1[dw*sets*ws + _set*ws + w]
				egalite = comparer(f0, f1)
				couleur = ('\033[42m' if egalite else '\033[101m')
				print(f"\033[93mdw={dw}\033[0m|\033[92mset={_set}\033[0m|\033[94m{w}\033[0m| {couleur} {float(f0)} --- {float(f1)} \033[0m")

				if not egalite:
					err = 1
	print("dw|set|w| ddS --- 1e10")
	if err == 1:
		ERR("Les valeurs sont pas les memes")

def dS(train, w0, plus0):
	train.null_D()
	train.restart()
	train.set_inputs()

	for s in range(train.sets):
		train._weight[s*train.mdl.weights + w0] += plus0

	train.calculer_dSdw()

	for s in range(train.sets):
		train._weight[s*train.mdl.weights + w0] -= plus0

	return [i for i in train._meand]

if __name__ == "__main__":
	for test in TEST_INSTS:
		print("\033[46m ====================================================================== \033[0m")
		print(f"\033[46m ========================\033[94m {test.__name__} \033[0m\033[46m============================= \033[0m")
		print("\033[46m ====================================================================== \033[0m")
		if test.calcule_dd:
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
				[i for i in range(mdl.weights)]
			)
			train.randomize(0)
			
			train.calculer_dSdwdw()
			dwdw = [i for i in train._dd_weight]

			_dd_weight = [0 for i in train._dd_weight]

			train.ddS_first_null()

			for dw in range(mdl.weights):
				_meand1 = dS(train, dw, 1e-5)
				_meand0 = dS(train, dw, 0)

				for _set in range(train.sets):
					for i in range(mdl.weights):
						wpos = dw*train.sets*mdl.weights + _set*mdl.weights + i
						_dd_weight[wpos] = 1e5*(_meand1[_set*mdl.weights + i] - _meand0[_set*mdl.weights + i])

				del _meand0, _meand1

			comparer_2_listes_3d(dwdw, _dd_weight, mdl.weights, train.sets, mdl.weights)
		else:
			print("\033[43m ========================================= \033[0m")
			print("\033[43m ========= Pas de ddF implémenté ========= \033[0m")
			print("\033[43m ========================================= \033[0m")