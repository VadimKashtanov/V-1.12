from kernel.py.etc import *

class Gtic:
	name = None
	ID = None

	params_names = None
	params_defaults = None

	#REQUIRE_DDF = None #il faudra ajouter ça
	MIN_ECHOPES = None

	def __call__(self):
		return self.gtic()

	def __init__(self, train):
		self.train = train
		self.config = train.config_gtic
		self.params = self.config.params
		self.ID = self.config._id

		self.train.sets = 'Pas calculé'

	def __getitem__(self, key):
		return self.params[key]#self.params_names[key]]

	def __setitem__(self, key, val):
		if not type(val) is int:
			raise Exception(f"Can't assign none positiv int to a parameter because kernel use `uint`. ({val} is not int type)")
		
		self.params[key] = val

	def check(self):
		assert not None in (self.name, self.ID, self.params_names, self.params_defaults)
		assert len(self.params_defaults) == len(self.params)
		assert all(type(elm) == int and elm >= 0 for elm in self.params)
		assert all(type(elm) == int and elm >= 0 for elm in self.params_defaults)

	def bins(self):
		bins = st_123
		bins += st.pack('II', self.ID, len(self.params))
		bins += st.pack('I'*len(self.params), *self.params)
		bins += st_123
		return bins

	def copy(self):
		return self.__class__(self.params)

	## Les fonctions

	def gtic(self):
		assert 0