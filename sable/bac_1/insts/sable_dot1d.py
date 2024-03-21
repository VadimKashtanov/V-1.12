from package.insts.simple_f_xy import Simple_F_XY_Inst

class SABLE_DOT1D(Simple_F_XY_Inst):
	# h = sum(gauss(x + p))
	# y = h * (x*w + b)

	name = "SABLE_DOT1D"

	#	A partire de X,Y on calcule combien il faut de W
	def calc_W(self, X, Y):
		W = X*Y + Y
		return W

	def f(self, x, w, start_seed=0):
		X,Y,W = self.params[0], self.params[1], self.W

		assert len(x) == X

		return [sum(x[j]*w[j*Y + i] for j in range(X))+w[X*Y+i] for i in range(Y)]