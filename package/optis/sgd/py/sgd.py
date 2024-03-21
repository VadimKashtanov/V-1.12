#langue ou tu ecrit le cuda dans un petit fichier, les constants et un peut de python-like et ca te genere tout un package.
#poincarelang (pcl) .pcl
#ca te genere le C/Cuda, le Python et autres langues si il le faut

from kernel.py.opti import *

class SGD(Opti):
	name = "SGD"

	description = '''w -= alpha * grad(w)'''

	params_names = ['alpha']
	params_defaults = [float_as_uint(0.1)]

	REQUIRE_DDF = False

	def opti(self):
		Ialpha, = self.params

		alpha = itof(Ialpha)

		mdl = self.train.model
		sets = self.train.sets
		ws = mdl.weights
		lines = self.train.data.lines

		for s in range(sets):
			for w in range(ws):
				self.train._weight[s*ws + w] -= alpha * self.train._meand[s*ws + w] / lines

	def __call__(self):
		return self.opti()