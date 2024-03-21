import struct as st

from kernel.py.etc import *

class Train:
	def __init__(self,
		model, data,
		config_score, config_opti, config_gtic,
		SCORES, OPTIS, GTICS,
		calcule_d:bool,
		calcule_dd:bool,
		dws:[int]):

		self.calcule_d = int(calcule_d)
		self.calcule_dd = int(calcule_dd)
		self.dws = dws #tous les weights qu'il faudra ddF. Car on peut ne pas tous les faire passer par le tableau hessien

		self.model = model
		self.mdl = model
		self.data = data
		self.sets = 0

		ws = model.weights
		total = model.total
		lines = data.lines
		locds = model.locds
		locd2s = model.locd2s

		self.config_score = config_score
		self.config_opti = config_opti
		self.config_gtic = config_gtic

		self.score = SCORES[config_score._id](self)
		self.opti = OPTIS[config_opti._id](self)
		self.gtic = GTICS[config_gtic._id](self)

		self.score.check()
		self.opti.check()
		self.gtic.check()

		if not type(self.sets) == int:
			ERR(f"self.set == {self.sets}")
		elif self.sets < 1:
			ERR(f"self.set == {self.sets}")

		sets = self.sets

		assert sets < 100

		self._weight = [0 for _ in range(sets * ws)]
		self._var = [0 for _ in range(sets * lines * total)]
		self._locd = None if not (calcule_d or calcule_dd) else [0 for _ in range(sets * lines * locds)]
		self._grad = [0 for _ in range(sets * lines * total)] #_d_var
		self._meand = [0 for _ in range(sets * ws)]		#_d_weight

		if calcule_dd:
			self._locd2 = [0 for _ in range(sets * lines * locd2s)]
			self._dd_weight = [0 for _ in range(len(dws) * sets * ws)]
			self._dd_var = [0 for _ in range(sets * lines * total)]
			self._dd_locd = [0 for _ in range(sets * lines * locds)]
			self._dd_grad = [0 for _ in range(sets * lines * total)]
			self._dd_meand = [0 for _ in range(sets * ws)]
		else:
			self._locd2 = None
			self._dd_weights = None
			self._dd_var = None
			self._dd_locd = None
			self._dd_grad = None
			self._dd_meand = None

		self.set_score = [None for _ in range(self.sets)]
		self.set_rank = [None for _ in range(self.sets)]
		self.set_podium = [None for _ in range(self.sets)]

	def check(self):
		if 0 in self._weight:
			print("[train.py] Warrning 0 in train._weight")

	#def __del__(self):
	#	del self._weight
	#	del self._var
	#	del self._locd
	#	del self._grad
	#	del self._meand

	####################

	def randomize(self, seed):
		ws = self.model.weights

		for s in range(self.sets):
			for w in range(ws):
				#	Il faut faire attention a ca, il faut que tout soit identique entre Cuda et Python
				wpos = s*ws + w
				self._weight[ wpos ] = pseudo_randomf_minus1_1(seed + wpos)

	def randomize_from_mdl(self, seed, k=0.1):
		ws = self.model.weights

		for s in range(self.sets):
			for w in range(ws):
				#	Il faut faire attention a ca, il faut que tout soit identique entre Cuda et Python
				wpos = s*ws + w
				self._weight[ wpos ] = self.model.w[w] + k*pseudo_randomf_minus1_1(seed + wpos)

	def replace_weight_set_by(self, _set, new_weigts):
		assert len(new_weigts) == self.model.weights
		for i in range(len(new_weigts)):
			self._weight[_set*self.model.weights + i] = new_weigts[i]

	####################

	def bins(self, arr:str):
		return st.pack('f'*len(eval(f'self.{arr}')), *eval(f'self.{arr}'))

	def bins_all(self):		
		return b''.join(st_123+self.bins(arr) for arr,_ in self.arrs_tenss.items())

	###############################

	def bin_score(self):
		return st.pack('f'*len(self.set_score), *self.set_score)

	def bin_rank(self):
		return st.pack('I'*len(self.set_rank), *self.set_rank)

	def bin_podium(self):
		return st.pack('I'*len(self.set_podium), *self.set_podium)
		
	#######################

	def set_inputs(self, batch=0):
		inputs = self.model.inputs
		lines = self.data.lines

		for i in range(inputs):
			for s in range(self.sets):
				for time in range(lines):
					self._var[time*self.sets*self.model.total + s*self.model.total + i] = self.data.input[batch*lines*inputs + time*inputs + i]

	###############################

	#def score(self, batch=0):
	#	self.score.score()

	def sort_scores(self):
		sorted_ = list(sorted(enumerate(self.set_score), key=lambda x:x[1]))
		self.set_podium = [_set for _set,score in sorted_]

		for rank,(_set,score) in enumerate(sorted_):
			self.set_rank[_set] = rank

	def loss(self, batch=0):
		self.score.loss(batch)

	def dloss(self, batch=0):
		self.score.dloss(batch)

	def ddloss(self, batch=0):
		self.score.ddloss(batch)

	###############################

	#def opti(self):
	#	self.opti.opti()

	#def gtic(self):
	#	self.gtic.gtic()

	################################################

	def calc_score_rapide(self, start_seed=0, batch=0):
		null(self._var)
		self.set_inputs(batch)
		self.forward(start_seed)
		self.score()
		return [s for s in self.set_score]

	def calculer_dw_1e5(self, start_seed=0, batch=0):
		null(self._meand)
		ws = self.mdl.weights
		for w in range(ws):
			for _set in range(self.sets): self._weight[_set*ws + w] += 1e-5
			set_score_1e5 = self.calc_score_rapide(batch=batch)

			for _set in range(self.sets): self._weight[_set*ws + w] -= 1e-5
			set_score = self.calc_score_rapide(batch=batch)

			for _set in range(self.sets):
				self._meand[_set*ws + w] = 1e5*(set_score_1e5[_set] - set_score[_set])

	def calculer_dwdw_1e10(self, start_seed=0, eta=1e-5, batch=0):
		null(self._dd_weight)
		ws = self.mdl.weights

		for dw in range(ws):
			for w in range(ws):
				#f(x+,y+)-f(x+)-f(y+)+f()
				for _set in range(self.sets): self._weight[_set*ws + w] += eta
				for _set in range(self.sets): self._weight[_set*ws + dw] += eta
				set_score_fxy = self.calc_score_rapide(batch=batch)

				for _set in range(self.sets): self._weight[_set*ws + dw] -= eta
				set_score_fx = self.calc_score_rapide(batch=batch)

				for _set in range(self.sets): self._weight[_set*ws + dw] += eta
				for _set in range(self.sets): self._weight[_set*ws + w] -= eta
				set_score_fy = self.calc_score_rapide(batch=batch)

				for _set in range(self.sets): self._weight[_set*ws + dw] -= eta
				set_score_f = self.calc_score_rapide(batch=batch)

				for _set in range(self.sets):
					pos = dw*self.sets*ws + _set*ws + w
					Sxy, Sx, Sy, S = set_score_fxy[_set], set_score_fx[_set], set_score_fy[_set], set_score_f[_set]
					self._dd_weight[pos] = (Sxy - Sx - Sy + S)/(eta*eta)

	################################################

	def forward(self, start_seed):
		mdl = self.mdl
		total, weights, locds = mdl.total, mdl.weights, mdl.locds
		for time in range(self.data.lines):
			for inst in self.model.insts:
				for _set in range(self.sets):
					inst.forward(start_seed,
						self.sets, total, weights, locds, 
						_set, time,
						self._weight, self._var, self._locd)

	def backward(self, start_seed):
		mdl = self.mdl
		total, weights, locds = mdl.total, mdl.weights, mdl.locds
		for time in list(range(self.data.lines))[::-1]:
			for inst in self.model.insts[::-1]:
				for _set in range(self.sets):
					inst.backward(start_seed,
						self.sets, total, weights, locds,
						_set, time,
						self._weight, self._var, self._locd, self._grad, self._meand)

	def calculer_dSdw(self, start_seed=0, batch=0):
		nulls(self._var, self._locd, self._grad, self._meand)

		self.set_inputs(batch)
		self.forward(start_seed)
		self.dloss()
		self.backward(start_seed)

	################################################

	def calculer_dwdw_1e5(self, start_seed=0, batch=0):
		self.null_dwdw()
		ws = self.mdl.weights
		for dw in range(ws):
			for _set in range(self.sets):
				self._weight[_set*ws + dw] += 1e-5
			self.calculer_dSdw(start_seed, batch)
			meands_1e5 = [m for m in self._meand]

			for _set in range(self.sets):
				self._weight[_set*ws + dw] -= 1e-5
			self.calculer_dSdw(start_seed, batch)
			meands = [m for m in self._meand]

			for _set in range(self.sets):
				for w in range(ws):
					self._dd_weight[dw*self.sets*ws + _set*ws + w] = 1e5*(meands_1e5[_set*ws + w] - meands[_set*ws + w])

	################################################

	def forward2(self, start_seed):
		mdl = self.mdl
		total, weights, locds, locds2 = mdl.total, mdl.weights, mdl.locds, mdl.locd2s
		for _set in range(self.sets):
			for time in range(self.data.lines):
				for inst in self.model.insts:
					inst.forward2(start_seed,
						self.sets, total, weights, locds, locds2,
						_set, time,
						self._weight, self._var,
						self._locd, self._grad, self._meand,
						self._locd2,
						self._dd_weight, self._dd_var,
						self._dd_locd, self._dd_grad, self._dd_meand)

	def backward2(self, start_seed):
		mdl = self.mdl
		total, weights, locds, locds2 = mdl.total, mdl.weights, mdl.locds, mdl.locd2s
		for _set in range(self.sets):
			for time in list(range(self.data.lines))[::-1]:
				for inst in self.model.insts[::-1]:
					inst.backward2(start_seed,
						self.sets, total, weights, locds, locds2,
						_set, time,
						self._weight, self._var,
						self._locd, self._grad, self._meand,
						self._dd_weight, self._dd_var,
						self._dd_locd, self._dd_grad, self._dd_meand)

	def backward_of_backward2(self, dw, start_seed):
		mdl = self.mdl
		total, weights, locds, locds2 = mdl.total, mdl.weights, mdl.locds, mdl.locd2s
		for _set in range(self.sets):
			for time in list(range(self.data.lines)):
				for inst in self.model.insts:
					inst.backward_of_backward2(start_seed,
						dw,
						self.sets, total, weights, locds, locds2,
						_set, time,
						self._weight, self._var,
						self._locd, self._grad, self._meand,
						self._locd2,
						self._dd_weight, self._dd_var,
						self._dd_locd, self._dd_grad, self._dd_meand)

	def backward_of_forward2(self, dw, start_seed):
		mdl = self.mdl
		total, weights, locds, locds2 = mdl.total, mdl.weights, mdl.locds, mdl.locd2s
		for _set in range(self.sets):
			for time in list(range(self.data.lines))[::-1]:
				for inst in self.model.insts[::-1]:
					inst.backward_of_forward2(start_seed,
						dw,
						self.sets, total, weights, locds, locds2,
						_set, time,
						self._weight, self._var,
						self._locd, self._grad, self._meand,
						self._locd2,
						self._dd_weight, self._dd_var,
						self._dd_locd, self._dd_grad, self._dd_meand)

	def calculer_dSdwdw(self, start_seed=0, batch=0):
		null(self._dd_weight)

		nulls(self._var, self._locd, self._locd2)
		nulls(self._grad, self._meand)
		
		self.set_inputs(batch)

		self.forward2(start_seed)
		self.dloss()
		self.backward2(start_seed)

		#	C'est comme si on derivait plusieurs loss functions ou la loss function c'est S = (meand[w] - 0), donc d(dS/meand[w])/dmeand[w] = 1
		for dw in self.dws:
			nulls(self._dd_var, self._dd_locd, self._dd_grad, self._dd_meand)
			
			#d(dS/dw)/d(dS/dw) = 1
			for _set in range(self.sets):
				self._dd_meand[_set*self.mdl.weights + self.dws[dw]] = 1

			self.backward_of_backward2(dw, start_seed)
			self.ddloss() #qui est en réalité d(dloss)/dvar
			self.backward_of_forward2(dw, start_seed)
			#train_deriv_set_input(train);	// pas utile pour dwidwj

	###########################################################################
	###########################################################################
	###########################################################################

	arrs_tenss = {
		'_weight' : [('sets', 'weights'), 'wsep'],
		'_var' : [('lines', 'sets', 'total'), 'vsep'],

		'_grad' : [('lines', 'sets', 'total'), 'vsep'],
		'_locd' : [('lines', 'sets', 'locds'), 'lsep'],
		'_meand' : [('sets', 'weights'), 'wsep'],

		'_dd_weight' : [('dws', 'sets', 'weights'), 'wsep'],
		'_dd_var' : [('lines', 'sets', 'total'), 'vsep'],
		'_dd_meand' : [('sets', 'weights'), 'wsep'],
		'_dd_grad' : [('lines', 'sets', 'total'), 'vsep'],
		'_dd_locd' : [('lines', 'sets', 'locds'), 'lsep'],
		'_locd2' : [('lines', 'sets', 'locd2s'), 'l2sep'],
	}

	def print(self, arr:str):
		mdl = self.mdl
		data = self.data
		dws, sets, weights, lines, total, locds, locds2 = len(self.dws), self.sets, mdl.weights, data.lines, mdl.total, mdl.locds, mdl.locds
		vsep, wsep, lsep, l2sep = mdl.vsep, mdl.wsep, mdl.lsep, mdl.l2sep

		assert arr in self.arrs_tenss.keys()

		#sinon pour n-dim, utiliser update(tenseur)
		if len(self.arrs_tenss[arr][0]) == 2:
			lbl0, lbl1 = self.arrs_tenss[arr][0]
			sep = self.arrs_tenss[arr][1]
			sep = eval(sep)
			labels, poss = list(zip(*sep))
			for s in range(eval(lbl0)):
				color = (93 if s % 2 else 96)

				print(f"\033[{color}m || \033[0m {lbl0} #{s}")
				for w in range(eval(lbl1)):
					if w in poss:
						print(f"\033[{color}m ||| --> \033[0m {labels[poss.index(w)]}")	
					print(f"\033[{color}m ||| {w}|\033[0m {eval(f'self.{arr}')[s*eval(lbl1) + w]}")

		elif len(self.arrs_tenss[arr][0]) == 3:
			lbl0, lbl1, lbl2 = self.arrs_tenss[arr][0]
			sep = self.arrs_tenss[arr][1]
			sep = eval(sep)
			labels, poss = list(zip(*sep))

			for l in range(eval(lbl0)):
				color_l = (92 if l % 2 else 91)

				if lbl0 != 'dws':
					print(f"\033[{color_l}m || \033[0m {lbl0} #{l}")
				else:
					print(f"\033[{color_l}m || \033[0m {lbl0} #{self.dws[l]} (self.dws[l])")

				for s in range(eval(lbl1)):
					color_s = (93 if s % 2 else 96)

					print(f"\033[{color_l}m ||\033[{color_s}m||\033[0m {lbl1} #{s}")
					for i in range(eval(lbl2)):
						if i in poss:
							print(f"\033[{color_l}m ||\033[{color_s}m|| --> \033[0m {labels[poss.index(i)]}")	
						print(f"\033[{color_l}m ||\033[{color_s}m|| {i}| \033[0m {eval(f'self.{arr}')[l*eval(lbl2)*eval(lbl1) + s*eval(lbl2) + i]}")
		else:
			ERR(f"len(self.arrs_tenss[arr][0]) == {len(self.arrs_tenss[arr][0])}, alors que ca doit etre 2 ou 3")

	def print_all(self):
		for arr, _ in self.arrs_tenss.items():
			print(f"========================== {arr.upper()} ===========================")
			self.print(arr)

	###########################################################################
	###########################################################################
	###########################################################################

	def compare(self, arr:str, array:[float], err=1e-8):
		mdl = self.mdl
		data = self.data
		dws, sets, weights, lines, total, locds, locds2 = len(self.dws), self.sets, mdl.weights, data.lines, mdl.total, mdl.locds, mdl.locds
		vsep, wsep, lsep, l2sep = mdl.vsep, mdl.wsep, mdl.lsep, mdl.l2sep

		assert arr in self.arrs_tenss.keys()

		#sinon pour n-dim, utiliser update(tenseur)
		if len(self.arrs_tenss[arr][0]) == 2:
			lbl0, lbl1 = self.arrs_tenss[arr][0]
			sep = self.arrs_tenss[arr][1]
			sep = eval(sep)
			labels, poss = list(zip(*sep))
			for s in range(eval(lbl0)):
				color = (93 if s % 2 else 96)

				print(f"\033[{color}m || \033[0m {lbl0} #{s}")
				for w in range(eval(lbl1)):
					if w in poss:
						print(f"\033[{color}m ||| --> \033[0m {labels[poss.index(w)]}")	
					pos = s*eval(lbl1) + w
					print(f"\033[{color}m ||| {w}|\033[0m {eval(f'self.{arr}')[pos]} -- {array[pos]}")

					diff = abs(eval(f'self.{arr}')[pos] - array[pos])
					if diff > err:
						raise Exception(f"Diff={diff} > err={err}")

		elif len(self.arrs_tenss[arr][0]) == 3:
			lbl0, lbl1, lbl2 = self.arrs_tenss[arr][0]
			sep = self.arrs_tenss[arr][1]
			sep = eval(sep)
			labels, poss = list(zip(*sep))

			for l in range(eval(lbl0)):
				color_l = (92 if l % 2 else 91)

				if lbl0 != 'dws':
					print(f"\033[{color_l}m || \033[0m {lbl0} #{l}")
				else:
					print(f"\033[{color_l}m || \033[0m {lbl0} #{self.dws[l]} (self.dws[l])")

				for s in range(eval(lbl1)):
					color_s = (93 if s % 2 else 96)

					print(f"\033[{color_l}m ||\033[{color_s}m||\033[0m {lbl1} #{s}")
					for i in range(eval(lbl2)):
						if i in poss:
							print(f"\033[{color_l}m ||\033[{color_s}m|| --> \033[0m {labels[poss.index(i)]}")	
						pos = l*eval(lbl2)*eval(lbl1) + s*eval(lbl2) + i
						print(f"\033[{color_l}m ||\033[{color_s}m|| {i}| \033[0m {eval(f'self.{arr}')[pos]} -- {array[pos]}")

						diff = abs(eval(f'self.{arr}')[pos] - array[pos])
						if diff > err:
							raise Exception(f"Diff={diff} > err={err}")
		else:
			ERR(f"len(self.arrs_tenss[arr][0]) == {len(self.arrs_tenss[arr][0])}, alors que ca doit etre 2 ou 3")

		print("         train -- compare_array")

	###################################