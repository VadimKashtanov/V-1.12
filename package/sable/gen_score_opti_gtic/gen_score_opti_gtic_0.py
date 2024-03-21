from kernel.py.config import Config
from kernel.py.etc import *

def gen_score_opti_gtic_0(self, mode, config):
	assert 'mdl' in dir(self)
	assert 'data' in dir(self)
	
	if mode == 'simple':
		config = {**{'sets':1, 'alpha':0.1}, **config}

		self.config_score = Config(0, [])
		self.config_opti = Config(0, [uint_alpha:=float_as_uint(config['alpha'])])
		self.config_gtic = Config(0, [sets:=config['sets']])
	elif mode == 'hessienne':
		pass
	else:
		raise Exception(f"Le mode {mode} n'existe pas.")