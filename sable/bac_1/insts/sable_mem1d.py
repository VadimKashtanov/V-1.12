from package.insts.simple_f_xhy import Simple_F_XHY_Inst
from kernel.py.etc import *

class SABLE_MEM1D(Simple_F_XHY_Inst):
	# h = sum(gauss(x + p))
	# y = h * (x*w + b)

	name = "SABLE_MEM1D"

	#	A partire de X,Y on calcule combien il faut de H et de W
	def HW(self, X, Y):
		assert X == Y
		H, W = X, 2*X
		return H, W

	def f(self, x, w, h, start_seed=0):
		X,Y,H,W = self.params[0], self.params[1], self.H, self.W

		#h = sum(gauss(x[i] + w[i] + h[i]) for i in range())
		h = [gauss(x[i]+w[i]) + gauss(h[i]+w[X+i]) for i in range(X)]
		y = [x[i]*h[i] for i in range(X)]
		
		return {
			'y' : y,
			'h' : h
		}