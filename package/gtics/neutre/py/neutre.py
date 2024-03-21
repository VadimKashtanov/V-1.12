from kernel.py.gtic import Gtic
from kernel.py.etc import *

class NEUTRE(Gtic):
	name = "NEUTRE"

	description = ''' Il ne fait rien que de mettre en place les sets'''

	params_names = ['sets']
	params_defaults = [1]

	def __init__(self, train):
		self.train = train
		self.config = train.config_gtic
		self.params = self.config.params
		self.ID = self.config._id

		self.construire_sets()

	def construire_sets(self):
		sets, = self.params
		self.train.sets = self.params[0]

	def gtic(self):
		sets, = self.params

		#	Il ne fait rien. C'est juste pour avoire un Gtic qui calcule les sets

	def __call__(self):
		return self.gtic()