from kernel.py.score import Score

class MEANSQUARED(Score):
	name = "MEANSQUARED"

	description = '''Loss(want,get) := 0.5 * (want - get)^2
d(Loss(want,get))/d(get) = 0.5*d(want^2 - 2*want*get + get^2)/d(get)
						 = 0.5*(0 - 2*want + 2*get)
						 = want - get
Score of a set = sum(output for lines for outputs) / (lines * outputs)
'''

	ALLOW_DDF = True

	params_names = []
	params_defaults = []

	def score(self, batch=0):
		train = self.train
		mdl = train.model
		data = train.data

		lines = data.lines
		sets = train.sets
		total = mdl.total
		outputs = data.outputs

		outstart = total - outputs

		scores = [0 for s in range(sets)]

		for s in range(sets):
			for l in range(lines):
				for o in range(outputs):
					pos = l*sets*total + s*total + outstart + o

					#	(get - want)**2/2
					scores[s] += .5*(train._var[pos] - data.output[batch*(lines*outputs) + l*outputs + o])**2
			#scores[s] /= lines * outputs #(il faut alors multiplier par 1/(l*o) dans les dloss et ddloss)

		self.train.set_score = scores

	def loss(self, batch=0):
		train = self.train
		mdl = train.model
		data = train.data

		lines = data.lines
		sets = train.sets
		total = mdl.total
		outputs = data.outputs

		outstart = total - outputs

		scores = [0 for s in range(sets)]

		for s in range(sets):
			for l in range(lines):
				for o in range(outputs):
					pos = l*sets*total + s*total + outstart + o

					#	(get - want)**2/2
					train._grad[pos] = ( 0.5*(train._var[pos] - data.output[batch*(lines*outputs) + l*outputs + o])**2)# / lines * outputs

	def dloss(self, batch=0):
		train = self.train
		mdl = train.model
		data = train.data

		lines = data.lines
		sets = train.sets
		total = mdl.total
		outputs = data.outputs

		outstart = total - outputs

		for l in range(lines):
			for s in range(sets):
				for o in range(outputs):
					#get - want
					#so : get -= want, but for clarity we will leav it as it is
					pos = (l*sets*total) + (s*total) + (outstart + o)
					train._grad[pos] = (train._var[pos] - train.data.output[batch*(lines*outputs) + (l*outputs) + o])# / lines * outputs

	def ddloss(self, batch=0):
		train = self.train
		mdl = train.model
		data = train.data

		lines = data.lines
		sets = train.sets
		total = mdl.total
		outputs = data.outputs

		outstart = total - outputs

		for l in range(lines):
			for s in range(sets):
				for o in range(outputs):
					#get - want
					#so : get -= want, but for clarity we will leav it as it is

					pos = (l*sets*total) + (s*total) + (outstart + o)
					#train._grad[pos] = train._var[pos] - train.data.output[batch*(lines*outputs) + (l*outputs) + o]

					train._dd_var[pos] += train._dd_grad[pos]# / lines * outputs