from package.insts.simple_f_xhy import Simple_F_XHY_Inst
from kernel.py.etc import *

class SABLE_PYRAMIDE(Simple_F_XHY_Inst):
	# h = sum(gauss(x + p))
	# y = h * (x*w + b)

	name = "SABLE_PYRAMIDE"

	#	A partire de X,Y on calcule combien il faut de H et de W
	def WH(self, X, Y):
		assert X == Y
		assert X > 1
		W, H = X*6 + (X-1)*3, X-1
		return W, H

	def f(self, x, w, h, start_seed=0):
		X,Y,H,W = self.params[0], self.params[1], self.H, self.W
		x = [0] + x + [0]
		h = [0] + h + [0]
		y = [w[i*6]*x[i] + x[i+1]*w[i*6+1] + x[i+2]*w[i*6+2] + w[i*6+3]*h[i] + w[i*6+4]*h[i+1] + w[i*6+5] for i in range(X)]
		x = x[1:-1]
		h = h[1:-1]
		h = [w[X*6 + i*3 + 0]*x[i] + w[X*6 + i*3 + 1]*x[i+1] + w[X*6 + i*3 + 2] for i in range(X-1)]
		
		return {
			'y' : y,
			'h' : h
		}