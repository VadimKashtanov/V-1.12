from kernel.py.inst import Inst

class BuildFromRequired(Inst):
	def build_from_required(self, required, inputs, istart, ystart, wstart, lstart, l2start):
		iter_required = iter(required)
		for i,req in enumerate(self.requiredposition):
			if req: self.params[i] = next(iter_required)
		self.check_input_output(inputs, required)
		self.params = self.setupparamsstackmodel(istart, ystart, wstart, lstart, l2start, required)
		return self