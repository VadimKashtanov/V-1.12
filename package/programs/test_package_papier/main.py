from package.test_package import TESTS, PAPIERS
from package.package import INSTS, SCORES, OPTIS, GTICS
from kernel.py.etc import *
from kernel.py.data import Data
from kernel.py.train import Train

class A: pass

def comparer(a, b, tolerance=0.001):
	return abs(a-b) < tolerance

def comparer_2_listes_2d(lst0, lst1, lignes, total):
	assert len(lst0) == len(lst1)
	err = 0
	for l in range(lignes):
		for i in range(total):
			egalite = comparer(lst0[l*total + i], lst1[l*total + i])
			couleur = ('\033[42m' if egalite else '\033[101m')
			print(f"\033[93m{l}\033[0m|\033[92m{i}\033[0m| {couleur} {float(lst0[l*total + i])} --- {float(lst1[l*total + i])} \033[0m")
			if not egalite:
				err = 1
	print("l|i|  Papier --- var")
	if err == 1:
		ERR("Les valeurs sont pas les memes")

def papier_insts():
	for papier, test_inst in zip(PAPIERS[0], TESTS[0]):
		if papier.PASSER == False:
			print(f" ### \033[95m{type(papier()).__name__}\033[0m ###")
			mdl = papier.mdl
			weight = papier.weight
			#var = papier.var
			inp = papier.inp

			total = mdl.total
			lignes = papier.lignes

			var = [0 for i in range(total * lignes)]
			for l in range(lignes):
				for i in range(len(inp[l])):
					var[l*total + i] = inp[l][i]

			for ligne in range(lignes):
				for inst in mdl.insts:
					inst.mdl(total, ligne, var, weight)

			comparer_2_listes_2d(var, papier.var, lignes, total)
		else:
			print(f"### Skip de {type(papier).__name__} ###")

def papier_scores():
	for papier, test in zip(PAPIERS[1], TESTS[1]):
		if papier.PASSER == False:
			print(f" ### \033[95m{type(papier()).__name__}\033[0m ###")
			mdl = A()
			mdl.weights = 0
			mdl.total = int(len(papier.want)/papier.lignes)
			mdl.inputs = 0
			mdl.locds = 0
			mdl.locd2s = 0
			data = Data(batchs=1, lines=papier.lignes, _input=[0 for l in range(papier.lignes)], output=papier.want)
			train = Train(mdl, data,
				test.config_score, test.config_opti, papier.config_gtic,
				SCORES, OPTIS, GTICS,
				test.calcule_d,
				test.calcule_dd,
				[]
			)
			train._var = papier.get

			lignes = papier.lignes
			total = mdl.total
			
			print("== Loss ==")
			train.loss()
			comparer_2_listes_2d(train._grad, papier._grad_loss, lignes, total)

			print("== Score ==")
			train.score()
			egalite = comparer(train.set_score[0], papier.score)
			couleur = ('\033[42m' if egalite else '\033[101m')
			print(f"Score : {couleur} {train.set_score[0]} --- {papier.score} \033[0m")
			print("         papier --- train")
			if not egalite: ERR("Les scores ne sont pas les memes")
			
			if test.calcule_d:
				print("== dLoss ==")
				train.dloss()
				comparer_2_listes_2d(train._grad, papier._grad_dloss, lignes, total)

			#grad = _d_var en réalité, et donc dd_var c'est 
			if (test.calcule_dd):
				print("== ddLoss ==")
				train._dd_grad = papier._dd_grad_ddloss
				train.ddloss()
				comparer_2_listes_2d(train._dd_var, papier._dd_grad_ddloss, lignes, total)

def papier_optis():
	for papier, test in zip(PAPIERS[2], TESTS[2]):
		if papier.PASSER == False:
			print(f" ### \033[95m{type(papier()).__name__}\033[0m ###")
			mdl = A()
			mdl.weights = len(papier.weight)
			mdl.weight = papier.weight
			mdl.total = 0
			mdl.inputs = 0
			mdl.locds = 0
			mdl.locd2s = 0
			data = Data(batchs=1, lines=1, _input=[0], output=[0])
			train = Train(mdl, data,
				test.config_score, test.config_opti, papier.config_gtic,
				SCORES, OPTIS, GTICS,
				test.calcule_d,
				test.calcule_dd,
				[]
			)

			train._weight = papier.weight
			for echope in range(test.MIN_ECHOPES):
				train._meand = papier.meand[echope]
				train._dd_weight = papier._dd_weight[echope]

				train.opti()
			
			comparer_2_listes_2d(train._weight, papier.apres_echopes, 1, mdl.weights)

def papier_gtics():
	print("Pour l'instant pas de papier pour Gtic.")
	print("Car Il faudrait ecrire tous les _d et les _dd_")

if __name__ == "__main__":
	print("=================== Comparaison des \033[91m INSTRUCTIONS \033[0m avec les donnee papier ======================")
	papier_insts()

	print("=================== Comparaison des \033[91m SCORES \033[0m avec les donnee papier ======================")
	papier_scores()

	print("=================== Comparaison des \033[91m OPTIS \033[0m avec les donnee papier ======================")
	papier_optis()

	print("=================== Comparaison des \033[91m GTICS \033[0m avec les donnee papier ======================")
	papier_gtics()

	print("============================================================================================= ")
	print("=============================      \033[92m Tout est bon \033[0m    ========================================== ")
	print("============================================================================================= ")