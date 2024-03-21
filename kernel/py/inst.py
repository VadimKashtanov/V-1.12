from kernel.py.etc import *

class Inst:
	name = None
	ID = None

	params_names = None

	ALLOW_DF = None
	ALLOW_DDF = None

	def __init__(self, preset=None):
		self.params_dict = {}
		self.params = []

		for i,param in enumerate(self.params_names):
			if preset != None:
				if type(preset[i]) is int and preset[i] >= 0:
					self.params_dict[param] = preset[i]
					self.params += [preset[i]]
				else:
					self.params_dict[param] = None
					self.params += [None]
			else:
				self.params_dict[param] = None
				self.params += [None]
		
		#p:(preset[p] if type(preset[p]) is int and preset[p] >= 0 else None) for i,p in enumerate(self.params_names)}

	def check(self):
		assert self.ID != None
		assert self.name != None
		assert self.params_names != None

		assert self.ALLOW_DF != None
		assert self.ALLOW_DDF != None

		#self.check_params()

	def check_params(self):
		assert 0
		
	def __getitem__(self, key):
		return self.params[key]#self.params_names[key]]

	def __setitem__(self, key, val):
		if not type(val) is int:
			raise Exception(f"Can't assign none positiv int to a parameter because kernel use `uint`. ({val} is not int type)")
		
		self.params[key] = val

	def bins(self):
		bins = st_123
		bins += st.pack('II', self.ID, len(self.params))
		bins += st.pack('I'*len(self.params), *self.params)
		bins += st_123
		return bins

	def load(self):
		read_123(bins)
		self.ID, params = read('II', bins)
		self.params = read('I'*params, bins)
		read_123(bins)
		return self

	def copy(self):
		return self.__class__(self.params)

	def check_model(self, insts_ids:[int], params:[[int]], this_inst_pos:int):
		raise Exception("Non Implémenté")

	def mdl(self, total, line, var, w):
		raise Exception("Non Implémenté")

	def forward(self, sets, total, ws, locds, _set, line, w, var, locd):
		raise Exception("Non Implémenté")

	def backward(self, sets, total, ws, locds, _set, line, w, var, locd, grad, meand):
		raise Exception("Non Implémenté")