from math import exp

gauss = lambda x: exp(-x*x) if -100 < x < 100 else 0.0

pseudo_random = lambda seed: (12769*((seed*seed%12345)+1) % 4877)/4877

C = 5
K = 5

def dot1d(x, w, h):
	N = len(x)
	ret = []
	for i in range(N):
		s = 0
		for j in range(N):
			h[i*N + j] = h[i*N + j]*gauss(w[i*1*N + j]) + gauss(w[i*2*N + j]*(x[i] + w[i*3*N + j]))
			s += x[i] * w[i*4*N + j] * (gauss(w[i*5*N + j]) + h[i*N + j])
		ret += [s + w[N*K*N + i]]
	return ret

class MEM:
	def restart(self):
		self.h = [0 for _ in range(len(self.h))]
	def __init__(self, *params):
		self.A, self.B = params

		self.h = [0 for _ in range(C*self.A*self.B)]
		
		self.w = [0 for i in range( C * (self.A * K*self.B + self.B))]

	def __call__(self, x, w):
		assert self.A == len(x)
		N = self.A

		W = K*N*N + N
		H = N*N
		y = x

		for _y in range(C):
			y = dot1d(x, w[_y*W:(_y+1)*W], self.h[_y*H:(_y+1)*H])

		return y