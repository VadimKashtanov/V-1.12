from package.insts.fast_model import *

def gen_mdl_0(self, mode, config):
	if mode == 'stack':
		assert type(config) in (list, tuple)
		assert all(type(elm) in (list, tuple) for elm in config)
		assert all(type(name)==str and type(params) in (list, tuple) for name,params in config)
		assert all(type(param) == int for name,params in config for param in params)

		self.mdl = Fast_NLayers_FeedForward_Mdl(
				[self.insts_dict[name], req_params]
			for name, req_params in config
		)
	else:
		raise Exception(f"Le mode {mode} n'existe pas.")