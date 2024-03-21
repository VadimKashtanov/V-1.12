from kernel.py.etc import *

import struct as st

class Config:
	def __init__(self, _id, params):
		self._id = _id
		self.params = params

	def bins(self):
		bins = st_123
		bins += st.pack('II', self._id, len(self.params))
		bins += st.pack('I'*len(self.params), *self.params)
		bins += st_123
		return bins

	def load(self, bins):
		read_123(bins)
		self._id, params = read('II', bins)
		self.params = read('I'*params, bins)
		read_123(bins)
		return self

	def print(self):
		print(f"Config._id = {self._id}")
		print(f"Params {self.params}")