from random import random

def optimiser_0(self, mode, config):
	if mode == 'opti':
		if self.train.opti.REQUIRE_DDF: self.dw_dwdw(batch=config['batch'])
		else: self.dw(batch=config['batch'])
		#
		self.train.opti()
	elif mode == 'random':
		assert self.train.sets == 1

		coef = config['coef']
		
		meilleurs_poids = [2*coef*(random()-.5) for _ in range(self.mdl.weights)]
		meilleur_score = self.compute_score(config['batch'])[0]

		nb = config['nb']
		for i in range(nb):
			self.train._weight = [2*coef*(random()-.5) for _ in range(self.mdl.weights)]
			score = self.compute_score(config['batch'])[0]
			if score < meilleur_score:
				meilleur_score = score
				meilleurs_poids = [p for p in self.train._weight]

		self.train._weight = meilleurs_poids
	else:
		raise Exception(f"Le mode {mode} n'existe pas.")