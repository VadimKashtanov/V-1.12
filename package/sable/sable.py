from os import listdir

def versions(obj, self, mode, config):
	versions = listdir(f'package/sable/{obj}')
	if '__pycache__' in versions:
		del versions[versions.index('__pycache__')]
	nb = max(map(lambda x:int(x.replace('.py','')[len(obj)+1:]), versions))
	obj_version = __import__(f'package.sable.{obj}.{obj}_{nb}')
	exec(f"from package.sable.{obj}.{obj}_{nb} import {obj}_{nb}")
	exec(f"{obj}_{nb}(self, mode, config)")

def gen_data(self, mode, config):
	versions('gen_data', self, mode, config)	#gen_data_{i}

def gen_mdl(self, mode, config):
	versions('gen_mdl', self, mode, config)		#gen_mdl_{i}

def gen_score_opti_gtic(self, mode, config):
	versions('gen_score_opti_gtic', self, mode, config) #gen_score_opti_gtic_{i}

def optimiser(self, mode, config):
	versions('optimiser', self, mode, config)		#optimiser_{i}

def dessinner(self, mode, config):
	versions('dessinner', self, mode, config)		#dessinner_{i}