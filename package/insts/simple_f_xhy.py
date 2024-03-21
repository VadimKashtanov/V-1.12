from kernel.py.etc import *
from kernel.py.inst import Inst

from package.insts.build_from_required import BuildFromRequired

'''

class Simple_F_XHY_inst(Simple_F_XHY_Inst):
	# h = sum(gauss(x + p))
	# y = h * (x*w + b)

	NAME = "Un nom"

	#	A partire de X,Y on calcule combien il faut de H et de W
	def WH(self, X, Y):
		assert X == Y
		H, W = 1, X+2*X
		return H, W

	def f(self, x, w, h):
		X,Y,H,W = self.X, self.Y, self.H, self.W

		h = sum(gauss(x + p) for i in range())
			
		return {
			'y' : [h * (x[i] for ) for i in range(X)]
			'h' : [h]
		}
'''

#Cette instruction permet d'avoire un h[-1] et un Y different du h[-1] (qui est la pour la memoire, ou qui peut etre tres bien nulle)

class Simple_F_XHY_Inst(BuildFromRequired):	#F(x, h[-1]) -> h, y
	#				F : X --> Y
	
	#	ça veut dire que le calcule sera uniquement celui du scalaire de la fonction Loss()
	ALLOW_DF = False
	ALLOW_DDF = False

	params_names = ['X', 'Y', 'istart', 'ystart', 'wstart']

	def copy(self):
		inst = self.__class__(self.params)
		inst.W, inst.H = self.W, self.H
		return inst

	def check_params(self):
		assert self.params[0]>0 and self.params[1]>0

	#W, H = None, None

	#def WH(self, X, Y):
	#	W, H = 0, 0
	#	return W, H

	def f(self, x, w, h, start_seed=0):
		return {
			'y' : [],
			'h' : []
		}

	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):

		X, Y, istart, ystart, wstart = self.params

		W, H = self.W, self.H

		#istart			ystart
		#  [x0,x1,x2] ... [h0,h1,y0,y1,y2,y3]

		yh = self.f(
			var[l*total + istart:l*total + istart + X],
			w[wstart:wstart + self.W],
			(var[(l-1)*total + ystart:(l-1)*total + ystart + self.H] if l > 0 else [0 for _ in range(self.H)])
		)

		for h in range(self.H):
			var[l*total + ystart + h] = yh['h'][h]

		for y in range(self.Y):
			var[l*total + ystart + self.H + y] = yh['y'][y]

	def forward(self, start_seed:int,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int,
		w:[float], var:[float], locd:[float]):

		X, Y, istart, ystart, wstart = self.params
		self.X, self.Y = X, Y

		H, W = self.H, self.W

		yh = self.f(
			var[line*sets*total + _set*total + istart:line*sets*total + _set*total + istart + X],
			w[_set*ws + wstart:_set*ws + wstart + W],
			(var[(line-1)*sets*total + _set*total + ystart:(line-1)*sets*total + _set*total + ystart + H] if line > 0 else [0 for _ in range(H)]),
			start_seed
		)

		for h in range(self.H):
			var[line*sets*total + _set*total + ystart + h] = yh['h'][h]

		for y in range(Y):
			var[line*sets*total + _set*total + ystart + self.H + y] = yh['y'][y]

	#### Build Stack Model  (Applications : "stack_model.py", )

	def return_iywll2_start(self):
		X, Y, istart, ystart, wstart = self.params
		return istart, ystart, wstart, None, None

	def relativ_ystart(self):
		X, Y, istart, ystart, wstart = self.params
		return self.H

	def buildstackmodel_vars(self):
		X, Y, istart, ystart, wstart = self.params
		return self.H + Y

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
		return [(f'{_id}.H [{self.name.lower()}]',stack_start), (f'{_id}.Y [{self.name.lower()}]',stack_start+self.H)]

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
		self.W, self.H = self.WH(X, Y)
		return X

	def check_input_output(self, last_vars, required):	#verifier que le vars du l'instruction precedente est de la meme taille 
		X, Y = required				#que le input pour cet instruction
		self.W, self.H = self.WH(X, Y)
		assert last_vars == self.need_inputs(required)