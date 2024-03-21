from package.insts.simple_f_xy import Simple_F_XY_Inst
from math import exp

class SABLE_REFLEXION(Simple_F_XY_Inst):
	# h = sum(gauss(x + p))
	# y = h * (x*w + b)

	name = "SABLE_REFLEXION"

	#	A partire de X,Y on calcule combien il faut de W
	def calc_W(self, X, Y):
		assert X == Y
		W = 10*X
		return W

	def f(self, x, w, start_seed=0):
		X,Y,W = self.params[0], self.params[1], self.W

		assert len(x) == X
		assert X == Y

		f = lambda x: 1 / (1 + exp(-x))

		return [
			sum(
				w[10*i + j*3]*f(x[i]*w[10*i + j*3 + 1] + w[10*i + j*3 + 2])
				for j in range(3)
			)+w[10*i + 3*3]
			
			for i in range(X)
		]