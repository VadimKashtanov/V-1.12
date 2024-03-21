import struct as st

#	/usr/local/lib/ssml

class Data:
	def __init__(self, batchs, lines, _input, output):
		self.batchs = batchs
		self.lines = lines
		self.inputs = int(len(_input)/(lines*batchs))
		self.outputs = int(len(output)/(lines*batchs))

		self.input = _input
		self.output = output

	def check(self):
		if not self.inputs * self.lines * self.batchs == len(self.input): ERR("")
		if not self.outputs * self.lines * self.batchs == len(self.output): ERR("")

	def bins(self):
		bins = st.pack('IIII', self.batchs, self.lines, self.inputs, self.outputs)

		bins += st.pack('f'*(self.batchs * self.lines * self.inputs), *self.input)
		bins += st.pack('f'*(self.batchs * self.lines * self.outputs), *self.output)

		return bins

	def load(self, bins):
		read = lambda size, a: (st.unpack(size, a[:st.calcsize(size)]), a[st.calcsize(size):])

		(self.batchs, self.lines, self.inputs, self.outputs), bins = read('IIII', bins)

		self.input, bins = read('f'*(self.batchs * self.lines * self.inputs), bins)
		self.output, bins = read('f'*(self.batchs * self.lines * self.outputs), bins)

		self.check()