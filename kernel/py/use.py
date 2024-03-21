from kernel.py.mdl import Mdl
import struct as st

class Use:
	def __init__(self, mdl, data):
		self.mdl = mdl
		self.data = data
		
		self._var = [0 for i in range(mdl.total * data.lines)]

		if self.mdl.w == None:
			self.mdl.w = [random() for _ in range(self.mdl.weights)]

	def __del__(self):
		del self._var
	
	def set_inputs(self, batch):
		for line in range(self.data.lines):
			for i in range(self.mdl.inputs):
				self._var[self.mdl.total * line + i] = self.data.input[batch*(self.data.lines * self.mdl.inputs) + line*self.mdl.inputs + i]

	def restart(self):
		for line in range(self.data.lines):
			for i in range(self.mdl.inputs):
				self._var[self.mdl.total * line + i] = 0 

	def forward(self):
		for line in range(self.data.lines):
			for inst in self.mdl.insts:
				inst.mdl(self.mdl.total, line, self._var, self.mdl.w)

	def bins(self):
		return st.pack('f'*len(self._var), *self._var)

	def print_vars(self):
		labels, poss = list(zip(*self.mdl.vsep))

		for l in range(self.data.lines):
			color_l = (92 if l % 2 else 91)

			print(f"\033[{color_l}m || \033[0m Line #{l}")
			for i in range(self.mdl.total):
				if i in poss:
					print(f"\033[{color_l}m || --> \033[0m {labels[poss.index(i)]}")	
				print(f"\033[{color_l}m || {i}| \033[0m {self._var[l*self.mdl.total + i]}")