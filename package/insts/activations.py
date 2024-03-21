from math import exp, tanh, pi, log

# 0 - Logistic
# 1 - Tanh
# 2 - Gauss
# 3 - ReLu
# 4 - Identity
# 5 - Tanh 0.8 (pour atan)
# 6 - SoftPlus

class Logistic:
	@staticmethod
	def f(x):
		if -100 < x < 100: return 1 / (1 + exp(-x))
		elif x >= 100: return 1.0
		else: return 0.0
	@staticmethod
	def df(x):
		f = Logistic.f(x)
		return f*(1-f)
	@staticmethod
	def ddf(x):
		f, df = Logistic.f(x), Logistic.df(x)
		return df*(1-2*f)

class Tanh:
	@staticmethod
	def f(x):
		return tanh(x)
	@staticmethod
	def df(x):
		f = Tanh.f(x)
		return 1 - f**2
	@staticmethod
	def ddf(x):
		f, df = Tanh.f(x), Tanh.df(x)
		return -2*f*df

class Gauss:
	@staticmethod
	def f(x):
		return exp(-x*x)
	@staticmethod
	def df(x):
		f = Gauss.f(x)
		return -2*x*f
	@staticmethod
	def ddf(x):
		f, df = Gauss.f(x), Gauss.df(x)
		return -2*f + 4*x*x*f

class ReLu:
	@staticmethod
	def f(x):
		return x * (x >= 0)
	@staticmethod
	def df(x):
		f = ReLu.f(x)
		return (x >= 0)
	@staticmethod
	def ddf(x):
		f, df = ReLu.f(x), ReLu.df(x)
		return 0

class Identite:
	@staticmethod
	def f(x):
		return x
	@staticmethod
	def df(x):
		f = Identite.f(x)
		return 1
	@staticmethod
	def ddf(x):
		f, df = Identite.f(x), Identite.df(x)
		return 0

class Tanh0_7:
	@staticmethod
	def f(x):
		return tanh(0.7*x)*pi/2
	@staticmethod
	def df(x):
		f = Tanh0_7.f(x)
		return (1 - f**2)*0.7*pi/2
	@staticmethod
	def ddf(x):
		f, df = Tanh0_7.f(x), Tanh0_7.df(x)
		return -2*f*(1 - f**2)*0.7*0.7*pi*pi/4

class SoftPlus:
	@staticmethod
	def f(x):
		return (log(1 + exp(x)) if -100 < x < 100 else (x if x > 100 else 1e-10))
	@staticmethod
	def df(x): #ln(1 + e^x) -> e^x / (1 + e^x) == e^(-x)/e^(-x) * e^(x) / (1 + e^x) == e^(x-x)/(e^(-x) + e^(x-x)) == 1 / (1 + e^(-x))
		f = SoftPlus.f(x)
		return Logistic.f(x)
	@staticmethod
	def ddf(x):
		f, df = SoftPlus.f(x), SoftPlus.df(x)
		return Logistic.df(x)

liste = [
	Logistic,
	Tanh,
	Gauss,
	ReLu,
	Identite,
	Tanh0_7,
	SoftPlus
]

activate = [
	classe.f for classe in liste
]

localderiv = [
	classe.df for classe in liste
]

local2deriv = [
	classe.ddf for classe in liste
]

activ_names = {
	"logistic" : 0,
	"tanh" : 1,
	"gauss" : 2,
	"relu" : 3,
	"identity" : 4, "id" : 4,
	"tanh0.7" : 5,
	"softplus" : 6
}