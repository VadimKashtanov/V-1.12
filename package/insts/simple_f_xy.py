from kernel.py.etc import *
from kernel.py.inst import Inst

from package.insts.build_from_required import BuildFromRequired

'''
class Simple_F_XY_inst(Simple_F_XHY_Inst):
	# h = sum(gauss(x + p))
	# y = h * (x*w + b)

	NAME = "Un nom"

	#	A partire de X,Y on calcule combien il faut de W
	def calc_W(self, X, Y):
		assert X == Y
		W = X+2*X
		return W

	def f(self, x, w):
		X,Y,W = self.X, self.Y self.W

		return [(x[i] for ) for i in range(X)]
'''

#Cette instruction permet d'avoire un h[-1] et un Y different du h[-1] (qui est la pour la memoire, ou qui peut etre tres bien nulle)

class Simple_F_XY_Inst(BuildFromRequired):	#F(x) -> y
	#				F : X --> Y
	
	#	ça veut dire que le calcule sera uniquement celui du scalaire de la fonction Loss()
	ALLOW_DF = False
	ALLOW_DDF = False

	params_names = ['X', 'Y', 'istart', 'ystart', 'wstart']

	#W = None

	def copy(self):
		inst = self.__class__(self.params)
		inst.W = self.W
		return inst

	def check_params(self):
		assert self.params[0]>0 and self.params[1]>0

	#def calc_W(self, X, Y):
	#	W = None
	#	return W

	def f(self, x, w, start_seed=0):
		return []

	#def __init__(self, params):
	#	super().__init__(params)
	#	self.W = self.calc_W(params[0], params[1])

	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):

		X, Y, istart, ystart, wstart = self.params

		W = self.W

		#istart			ystart
		#  [x0,x1,x2] ... [h0,h1,y0,y1,y2,y3]

		resultat = self.f(
			var[l*total + istart:l*total + istart + X],
			w[wstart:wstart + self.W],
		)

		for y in range(Y):
			var[l*total + ystart + y] = resultat[y]

	def forward(self, start_seed:int,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int,
		w:[float], var:[float], locd:[float]):

		X, Y, istart, ystart, wstart = self.params

		W = self.W

		resultat = self.f(
			var[line*sets*total + _set*total + istart:line*sets*total + _set*total + istart + X],
			w[_set*ws + wstart:_set*ws + wstart + W],
			start_seed
		)

		for y in range(Y):
			var[line*sets*total + _set*total + ystart + y] = resultat[y]

	#### Build Stack Model  (Applications : "stack_model.py", )

	def return_iywll2_start(self):
		X, Y, istart, ystart, wstart = self.params
		return istart, ystart, wstart, None, None

	def relativ_ystart(self):
		X, Y, istart, ystart, wstart = self.params
		return 0

	def buildstackmodel_vars(self):
		X, Y, istart, ystart, wstart = self.params
		return Y

	def buildstackmodel_weights(self):
		X, Y, istart, ystart, wstart = self.params
		return self.W

	def buildstackmodel_locds(self):
		X, Y, istart, ystart, wstart = self.params
		return 0

	def buildstackmodel_locd2s(self):
		X, Y, istart, ystart, wstart = self.params
		return 0

	#### Labels Stack Model  (Applications : "stack_model.py", )

	def labelstackmodel_vars(self, _id, stack_start):
		X, Y, istart, ystart, wstart = self.params
		return [(f'{_id}.Y [{self.name.lower()}]',stack_start)]

	def labelstackmodel_weights(self, _id, stack_start):
		X, Y, istart, ystart, wstart = self.params
		return [(f'{_id}.W [{self.name.lower()}]',stack_start)]

	def labelstackmodel_locds(self, _id, stack_start):
		X, Y, istart, ystart, wstart = self.params
		return []

	def labelstackmodel_locd2s(self, _id, stack_start):
		X, Y, istart, ystart, wstart = self.params
		return []

	### Setput Params Stack Model  (Applications : "stack_model.py", )

	requiredforsetupparams = "X", "Y"		#build vars,weights and locd have to ask only for thoses params

	requiredposition = 1,1,0,0,0 #1,1,1 == Ax,Yx,activ

	params_defaults = {}

	def setupparamsstackmodel(self, istart, ystart, wstart, lstart, l2start, required):
		X, Y = required
		return X, Y, istart, ystart, wstart

	### Check Params Input output

	def need_inputs(self, required):
		X, Y = required
		self.W = self.calc_W(X, Y)
		return X

	def check_input_output(self, last_vars, required):	#verifier que le vars du l'instruction precedente est de la meme taille 
		X, Y = required				#que le input pour cet instruction
		self.W = self.calc_W(X, Y)
		assert last_vars == self.need_inputs(required)