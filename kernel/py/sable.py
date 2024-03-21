from kernel.py.train import Train

from package.sable.sable import gen_data, gen_mdl, gen_score_opti_gtic, optimiser, dessinner

class Sable:
	def __init__(self, tous_les_objets):
		self.insts = tous_les_objets['insts']
		self.scores = tous_les_objets['scores']
		self.optis = tous_les_objets['optis']
		self.gtics = tous_les_objets['gtics']
		
		for elm in self.insts:
			elm().check()

		self.insts_dict = {inst.name.lower() : inst for inst in self.insts}
		self.scores_dict = {score.name.lower() : score for score in self.scores}
		self.optis_dict = {opti.name.lower() : opti for opti in self.optis}
		self.gtics_dict = {gtic.name.lower() : gtic for gtic in self.gtics}

	############ Generation ############

	def gen_data(self, mode, config):
		gen_data(self, mode, config)

	def gen_mdl(self, mode, config):
		gen_mdl(self, mode, config)

	def gen_score_opti_gtic(self, mode, config):
		gen_score_opti_gtic(self, mode, config)

	def gen_train(self):
		calcule_d = self.mdl.allow_df()
		calcule_dd = all(
			[self.mdl.allow_ddf(),
			self.scores[self.config_score._id].ALLOW_DDF,
			self.optis[self.config_score._id].REQUIRE_DDF]
		)
		
		dws = list(range(self.mdl.weights))
		
		if 'train' in dir(self):
			del self.train

		self.train = Train(
			self.mdl, self.data,
			###################
			self.config_score,
			self.config_opti,
			self.config_gtic,
			###################
			self.scores,
			self.optis,
			self.gtics,
			###################
			calcule_d:=calcule_d,
			calcule_dd:=calcule_dd,
			dws:=dws
		)
		self.train.randomize_from_mdl(0, k=0.01)
		self.train.replace_weight_set_by(0, self.mdl.w)

		self.scores = []

	############# Fonctions ############

	def compute_score(self, batch):
		return self.train.calc_score_rapide(batch)

	def dw(self, start_seed=0, batch=0):
		if self.train.calcule_d:
			self.train.calculer_dSdw(start_seed, batch)
		else:
			self.train.calculer_dw_1e5(start_seed, batch)

	def dwdw(self, start_seed=0, batch=0):
		if self.train.calcule_dd:
			self.train.calculer_dSdwdw(start_seed, batch)
		elif self.train.calcule_d:
			self.train.calculer_dwdw_1e5(start_seed, batch)
		else:
			self.train.calculer_dwdw_1e10(start_seed, batch)

	def dw_dwdw(self, start_seed=0, batch=0):
		if self.train.calcule_dd:
			self.train.calculer_dSdwdw(start_seed, batch)
		elif self.train.calcule_d:
			self.train.calculer_dwdw_1e5(start_seed, batch)
			self.calculer_dSdw(start_seed, batch)
		else:
			self.train.calculer_dwdw_1e10(start_seed, batch)
			self.train.calculer_dw_1e5(start_seed, batch)
	
	############# Apprendre ############

	def optimiser(self, mode, config):
		#utilise gtic & opti
		assert 'batch' in config.keys()
		optimiser(self, mode, config)
		self.scores += [self.compute_score(config['batch'])]

	def restart_session(self):
		self.scores = []

	############# Dessinner ############

	def dessinner(self, mode, config):
		dessinner(self, mode, config)