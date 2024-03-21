from kernel.py.etc import *
from kernel.py.use import Use

def dessinner_0(self, mode, config):
	if mode == 'scores':
		for i in range(self.train.sets):
			print(f"\033[43m=============== Set #{i} ===============\033[0m")
			print("\033[92mEvolution du score de ce set en fonction du temps.\033[0m")
			term_plot([scores[i] for scores in self.scores])
	elif mode == 'serie':
		if type(config) != dict: config = {}
		config = {**{'batch':None}, **config}
		use = Use(self.train.mdl, self.train.data)

		feuilles = []
		batchs = (self.data.batchs if config['batch'] == None else 1)
		for batch in range(batchs):
			if config['batch'] != None: batch = config['batch']
			feuilles += [[]]
			for s in range(self.train.sets):
				print(f"========= Set #{s} =========")
				use.mdl.w = self.train._weight[s*use.mdl.weights:(s+1)*use.mdl.weights]
				use.set_inputs(batch)
				use.forward()
				X = use.data.inputs
				assert X == use.data.outputs
				total = use.mdl.total
				bsize = use.data.lines * X
				inp = [i for i in use._var[:X]]
				get = [[i for i in use._var[(l+1)*total - X:(l+1)*total]] for l in range(use.data.lines)]
				want = [[i for i in use.data.output[batch*bsize + l*X:batch*bsize + (l+1)*X]] for l in range(use.data.lines)]

				feuilles[-1] += [(inp, get, want)]

		print("import matplotlib")
		import matplotlib.pyplot as plt

		print("fig, ax = plt.subplots(batchs, self.train.sets)")
		#fig, axss = plt.subplots(batchs, self.train.sets)
		fig, axss = plt.subplots(self.train.sets, batchs)
		for batch in range(batchs):
			for s, (inp, get, want) in enumerate(feuilles[batch]):
				#axs = (axss if batchs==1 else axss[batch])
				#ax = (axs if self.train.sets==1 else axs[s])
				axs = (axss if self.train.sets==1 else axss[s])
				ax = (axs if batchs==1 else axs[batch])
				
				ligne_totale = inp
				for l in range(use.data.lines): ligne_totale += want[l]
				ax.plot([i for i in range((1+use.data.lines)*X)], ligne_totale, label='want')

				#	On nomme les premieres
				ax.plot([X + i for i in range(-1,X)], [ligne_totale[X-1]]+get[0], 'r-', label='get')
				
				#	Puis les autres predictions
				for l in range(1, use.data.lines):
					ax.plot([(1+l)*X + i for i in range(-1,X)], [ligne_totale[(l+1)*X-1]]+get[l], 'r-')

				ax.set_title(f"Batch={batch}, Set={s}")
				
		plt.legend()
		plt.show()
	else:
		raise Exception(f"Le mode {mode} n'existe pas.")