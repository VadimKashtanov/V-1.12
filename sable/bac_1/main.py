#! /usr/bin/python3

from kernel.py.sable import Sable

from random import random

#######################################################

def main_0(tous_les_objets):
	X = 5

	sable = Sable(tous_les_objets)
	sable.gen_mdl(
		mode='stack',
		config=[
			#('sable_pyramide', [Ax:=X, Yx:=X]),#, activ:=4]),
			('dot1d', [Ax:=X, Yx:=10, activ:=0]),
			('sable_dot1d', [Ax:=10, Yx:=X])
			#('sable_pyramide', [Ax:=X, Yx:=X]),#, activ:=4]),
		]
	)
	sable.mdl.w = [random()-.5 for _ in range(sable.mdl.weights)]
	X = 5
	lignes = 4
	batchs = 3
	s = 0
	serie = [s:=(s+random()-.5) for _ in range((X)*(lignes*batchs + 1))]
	sable.gen_data(
		mode='serie',
		config={
			'serie' : serie,
			'X' : X,
			'batchs' : batchs
		}
	)
	sable.gen_score_opti_gtic(
		mode='simple',
		config={
			'sets' : 1,
			'alpha' : 0.1
		}
	)
	sable.gen_train()
	sable.train.check()

	sable.mdl.print()

	for i in range(20):
		#if i % 10 == 0: sable.restart_session()
		sable.optimiser('opti', {'batch':i%3})

	sable.dessinner('scores', None)
	
	sable.dessinner('serie', None)

	#sable.restart_session()
	#sable.optimiser('random', {'batch':0, nb':10000, 'coef':3})
	#sable.dessinner('scores', None)

########################################################

def main(tous_les_objets):
	main_0(tous_les_objets)