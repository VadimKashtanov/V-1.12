import struct as st

from kernel.py.inst import Inst
from random import random

from kernel.py.etc import *

class Mdl:
	def __init__(self, insts, inputs, outputs, _vars, weights, w, locds, locd2s, vsep, wsep, lsep, l2sep):
		#On verifie si tout ce qu'il faut est dans `global`
		#assert all(elm in globals() for elm in ('INSTS', 'SCORES', 'OPTIS', 'GTICS'))
		
		for inst in insts:
			inst.check()
			inst.check_params()

		self.insts = insts

		self.inputs = inputs
		self.outputs = outputs

		#self.vars = _vars
		self.weights = weights
		self.w = w
		if type(self.w) == list:
			if len(self.w) != weights:
				raise Exception(f"Dans Mdl() w=None ou alors weights==len(w). En l'occurrence len(self.w)={len(self.w)} et weights={weights}")
		elif self.w != None:
			raise Exception(f"Dans Mdl() w=None ou alors weights==len(w). En l'occurrence type(self.w)={type(self.w)}")

		self.locds = locds
		self.locd2s = locd2s

		self.total = _vars + self.inputs

		#	Separators
		self.vsep = vsep
		self.wsep = wsep
		self.lsep = lsep
		self.l2sep = l2sep

	def allow_df(self):
		return all(inst.ALLOW_DF==True for inst in self.insts)

	def allow_ddf(self):
		ddf = all(inst.ALLOW_DDF==True for inst in self.insts)
		if ddf and not self.allow_df(): ERR("Si le model peut calculer DDF, il se doit de pouvoire calculer DF.")
		return ddf

	def build_weights(self):
		#	J'ai fait ça car les models peuvent avoire des millions de weights, ou beaucoup plus, ou avoire des erreurs de tailles
		#	Et donc afin de ne pas surcharger la phase pré-utilisation
		self.w = [2*random()-1 for _ in range(self.weights)]
		return self

	def copy(self):
		return Mdl(
			[i.copy() for i in self.insts], 
			self.inputs, self.outputs, 
			self.total-self.outputs, 
			self.weights, ([a for a in self.w] if type(self.w)==list else None),
			self.locds, self.locd2s,
			[(k,v) for k,v in self.vsep], [(k,v) for k,v in self.wsep], [(k,v) for k,v in self.lsep], [(k,v) for k,v in self.l2sep])

	def check(self):
		params = [inst.params for inst in self.insts]
		insts_ids = [inst.ID for inst in self.insts]

		for i,inst in enumerate(self.insts):
			inst.check()
			inst.check_model(insts_ids=insts_ids, params=params, this_inst_pos=i)

		assert self.vsep != None
		assert self.wsep != None
		assert self.lsep != None
		assert self.lsep != None

		for i in range(len(self.vsep)):
			assert self.vsep[i][1] < self.total

		for i in range(len(self.wsep)):
			assert self.wsep[i][1] < self.weights

		for i in range(len(self.lsep)):
			assert self.lsep[i][1] < self.locds

		for i in range(len(self.lsep)):
			assert self.l2sep[i][1] < self.locd2s

	def print(self):
		self.print_insts()
		print(f"""weights = {self.weights}
total = {self.total}
locds = {self.locds}
locd2s = {self.locd2s}""")

	def print_insts(self):
		for i,inst in enumerate(self.insts):
			print(f"{i}| {inst.name}   " + ' '.join(inst.params_names[i] + '=' + str(inst.params[i]) for i in range(len(inst.params))))

	def print_weights(self):
		ws = self.weights

		labels, poss = list(zip(*self.wsep))
		
		for w in range(ws):
			if w in poss:
				print(f" {labels[poss.index(w)]}")	
			print(f"{w}|  {self.w[w]}")

	def bins(self):
		bins = st_123
		bins += st.pack('I', len(self.insts))

		for inst in self.insts:
			bins += inst.bins()

		bins += st.pack('IIIIII', self.inputs, self.outputs, (_vars:=(self.total-self.inputs)), self.weights, self.locds, self.locd2s)

		if self.w == [] or self.w == None:
			for i in range(self.weights):
				bins += st.pack('f', random())
		elif type(self.w) == list:
			if len(self.w) == self.weights:
				bins += st.pack('f'*self.weights, *self.w)
			else:
				raise Exception("len(Mdl.w) == self.weights")
		else:
			raise Exception("Mdl.w est ni [] ni None")

		#	Separators
		for sep in self.vsep,self.wsep,self.lsep,self.l2sep:
			bins += st.pack('I', len(sep))
			for lbl,pos in sep:
				bins += st_123
				bins += st.pack('I', len(lbl))
				bins += lbl.encode()
				bins += st.pack('I', pos)
				bins += st_123

		return bins

	def load(self, bins):
		lire_123(bins)

		_len = read('I', bins)
		self.insts = [Inst().load(bins) for _ in range(_len)]
		self.insts = [INSTS[inst.ID](inst.params) for inst in self.insts]

		self.inputs, self.outputs, _vars, self.weights, self.locds, self.locd2s = read('IIIIII', bins)
		self.total = self.inputs + _vars

		self.w = read('f'*self.weights, bins)
		self.weights = len(self.w)

		#	Separators
		for sep in self.vsep,self.wsep,self.lsep:
			_seps = read('I', bins)
			for _ in range(_seps):
				lire_123(bins)
				lbl = read_str(bins)
				pos = read('I', bins)
				lire_123(bins)
				sep += [(lbl, pos)]

		self.check()
		return self