#	La j'implemente quand on fait : `for dw in dws: dd_algo()`
#	Mais je peux tres bien calculer tous les d(ds/dw)/d[] en un tour.
#	Deja ca require d'alloc dws* plus de memoire pour les _dd_var, _dd_grad ...
#	Sauf que dans les backward_of_backward2/backward_of_forward2 il faudra faire un `for dw in dws`.
#	Prcq cette matrice hessienne c'est comme si on avait plusieurs loss functions sur un meme modele qui
#	partent juste avec des valeurs differentes a chaque fois (le _dd_meand[dw]=1).
#	A moins de decortiquer totalement les forward2, backward2 en mode schema comme j'avais fais sur le papier
#
#	Cette methode est "aveugle". On backward_backward2_forward2 pour chaque weight sans savoire ou leur influence commence et termine
#	On calculera beaucoup d'innutile. Mais c'est plus rapide que juste (df(x+1e-5)-df(x))/1e-5, et pas juste d'un coeficient

from math import tanh
from random import random, randint, seed
from os import system
from time import time

def inverser(M):
	n = int(len(M)**.5)
	assert len(M) == n*n

	a = [elm for elm in M]

	_inverse = [int(i==j) for i in range(n) for j in range(n)]

	for L in range(n):
		for y in range(n):
			if y == L: continue #skip cette etape
			
			if a[L*n + L] == 0:
				return False

			coef = a[y*n + L]/a[L*n + L]
			#soustraire la ligne L a la ligne y
			for k in range(n):
				a[y*n + k] -= a[L*n + k]*coef
				_inverse[y*n + k] -= _inverse[L*n + k]*coef
			#print(a, _inverse)
	for L in range(n):
		coef = a[L*n + L]
		if coef == 0:
			return False
			
		for i in range(n):
			a[L*n + i] /= coef
			_inverse[L*n + i] /= coef
	return _inverse

def null_arrs(*liste):
	for arr in liste:
		for i in range(len(arr)):
			arr[i] = 0

def copy(arr):
	return [i for i in arr]

def print_matrice(mat2d):
	for i in range(len(mat2d)):
		for j in range(len(mat2d[0])):
			dxdy = mat2d[i][j]
			if dxdy < 0: system('printf "-"')
			else: system('printf " "')

			system(f'printf "%e, " {abs(dxdy)}')
		print("")

##############################################################################################

#	Structure du modele
# 2 inputs
# dot1d 2 -> 3
# dot1d 3 -> 1
# 1 outputs dans L(want,get) = (want-get)**2/2

suite = [5,10,3]#[1, 2, 1]

config = [
	#Ax, Yx
	[suite[i],suite[i+1]] for i in range(len(suite)-1)
]

assert all(config[i][-1]==config[i+1][0] for i in range(len(config)-1))

params = []
inputs, outputs = config[0][0], config[-1][-1]
total = 0
weights = 0
locds = 0
locds2 = 0

for Ax, Yx in config:
	total += Ax
	#			Ax, Yx, istart,  ystart,  wstart, lstart, l2start
	params += [[Ax, Yx, total-Ax, total, weights, locds, locds2]]
	weights += Ax*Yx + Yx
	locds += Yx #f'
	locds2 += Yx #f''

total += Yx

seed(0)

data = [
	[1000*random() for i in range(inputs)], [random() for i in range(outputs)]
]

w = [2*(random()-0.5) for i in range(weights)]
var = [0 for i in range(total)]

locd = [0 for i in range(locds)]
grad = [0 for i in range(total)]
meand = [0 for i in range(weights)]

#On ignore si les gradient sont nulls, on calcule le tableau dxdy directement
locd2 = [0 for i in range(locds2)]																	#	_locd2
d2var = [0 for i in range(weights * total)]	#d(dL/dw)/dx 											#	_dd_var
grad2 = [0 for i in range(weights * total)] #d(dL/dw)/(dL/dx) 										#	_dd_grad
meand2 = [0 for i in range(weights * weights)]	#d(dL/dw)/dw 										#	_dd_weight
dlocd = [0 for i in range(weights * locds)]	#d(dL/dw)/d(locd)										#	_dd_locd
Dmeand = [0 for i in range(weights * weights)] #d(dL/dwi)/d(dL/dwj) 								#	_dd_meand
#Il n'y a pas de d(dL/dwi)/d(dL/dwj) car il n'auront pas d'impacte (on va voire ca plus tard)

def print_state():
	arrs = ('w', 'var', 'locd', 'grad', 'meand', 'locd2', 'd2var', 'grad2', 'meand2', 'dlocd', 'Dmeand')
	for arr in arrs:
		print(arr, eval(arr))

#tanh(x)
#tanh'(x) = 1 - tanh(x)**2
#tanh''(x) = - 2 * tanh(x) * (1 - tanh(x)**2)  = 2*tanh(x)*(tanh(x) **2 - 1)

##############################################################################################

def dot1d_use(params):
	Ax, Yx, istart, ystart, wstart, lstart, l2start = params

	for y in range(Yx):
		s = w[wstart + Ax*Yx + y]
		for k in range(Ax):
			s += var[istart + k] * w[wstart + k*Yx + y]
		var[ystart + y] = tanh(s)

##############################################################################################

def dot1d_grad_forward(params):
	Ax, Yx, istart, ystart, wstart, lstart, l2start = params

	for y in range(Yx):
		s = w[wstart + Ax*Yx + y]
		for k in range(Ax):
			s += var[istart + k] * w[wstart + k*Yx + y]
		var[ystart + y] = tanh(s)
		locd[lstart + y] = 1 - tanh(s)**2

def dot1d_grad_backward(params):
	Ax, Yx, istart, ystart, wstart, lstart, l2start = params

	for y in range(Yx):
		dlds = grad[ystart + y] * locd[lstart + y]

		meand[wstart + Ax*Yx + y] += dlds

		for k in range(Ax):
			#s += var[istart + k] * w[wstart + k*Yx + y]
			grad[istart + k] += dlds * w[wstart + k*Yx + y]
			meand[wstart + k*Yx + y] += dlds * var[istart + k]

##############################################################################################

def dot1d_grad2_f2(params):
	Ax, Yx, istart, ystart, wstart, lstart, l2start = params

	for y in range(Yx):
		s = w[wstart + Ax*Yx + y]
		for k in range(Ax):
			s += var[istart + k] * w[wstart + k*Yx + y]
		var[ystart + y] = tanh(s)
		locd[lstart + y] = 1 - tanh(s)**2
		locd2[lstart + y] = -2*tanh(s)*(1 - tanh(s)**2)

def dot1d_grad2_b2(params):
	Ax, Yx, istart, ystart, wstart, lstart, l2start = params

	for y in range(Yx):
		dlds = grad[ystart + y] * locd[lstart + y]

		meand[wstart + Ax*Yx + y] += dlds

		for k in range(Ax):
			#s += var[istart + k] * w[wstart + k*Yx + y]
			grad[istart + k] += dlds * w[wstart + k*Yx + y]
			meand[wstart + k*Yx + y] += dlds * var[istart + k]

def dot1d_grad2_bb2(params):
	Ax, Yx, istart, ystart, wstart, lstart, l2start = params

	for dw in range(weights):
		for y in range(Yx):
			dlds = grad[ystart + y] * locd[lstart + y]

			#meand[wstart + Ax*Yx + y] += dlds

			Ddlds = 0

			for k in range(Ax):
				#grad[istart + k] += dlds * w[wstart + k*Yx + y]
				if grad2[dw*total + istart + k] != 0:
					Ddlds += grad2[dw*total + istart + k]* w[wstart + k*Yx + y]
					meand2[dw*weights + wstart + k*Yx + y] += grad2[dw*total + istart + k] * dlds 
				#meand[wstart + k*Yx + y] += dlds * var[istart + k]
				if Dmeand[dw*weights + wstart + k*Yx + y] != 0:
					Ddlds += var[istart + k] * Dmeand[dw*weights + wstart + k*Yx + y]
					d2var[dw*total + istart + k] += dlds * Dmeand[dw*weights + wstart + k*Yx + y]

			Ddlds += Dmeand[dw*weights + wstart + Ax*Yx + y]
			
			if Ddlds != 0:
				grad2[dw*total + ystart + y] += locd[lstart + y] * Ddlds
				dlocd[dw*locds + lstart + y] += grad[ystart + y] * Ddlds

def dot1d_grad2_bf2(params):
	Ax, Yx, istart, ystart, wstart, lstart, l2start = params

	for dw in range(len(w)):
		for y in range(Yx):
			#s = w[wstart + Ax*Yx + y]

			#for k in range(Ax):
			#	s += var[istart + k] * w[wstart + k*Yx + y]
			
			#var[ystart + y] = tanh(s)
			#locd[lstart + y] = 1 - tanh(s)**2
			#locd2[lstart + y] = -2*tanh(s)*(1 - tanh(s)**2)

			ds = 0
			ds += locd2[lstart + y] * dlocd[dw*locds + lstart + y]
			ds += locd[lstart + y] * d2var[dw*total + ystart + y]
			if ds != 0:
				for k in range(Ax):
					#s += var[istart + k] * w[wstart + k*Yx + y]
					d2var[dw*total + istart + k] += ds * w[wstart + k*Yx + y]
					meand2[dw*weights + wstart + k*Yx + y] += ds * var[istart + k]

			meand2[dw*weights + wstart + Ax*Yx + y] += ds

##############################################################################################

def set_inp():
	inp, out = data
	for i in range(len(inp)):
		var[i] = inp[i]

def deriv_set_inp():
	inp, out = data
	for w in range(weights):
		for i in range(len(inp)):
			d2var[w*total + 0 + i] += 1

def L():
	inp, out = data
	return sum((var[-(1+i)] - out[-(i+1)])**2/2 for i in range(len(out)))

def dL():
	inp, out = data
	for i in range(len(out)):
		grad[-(1+i)] = (var[-(1+i)] - out[-(1+i)])

def ddL():
	inp, out = data
	for w in range(weights):
		for i in range(len(out)):
			#grad2[w*total + total - outputs + i] += 1
			d2var[w*total + total - outputs + i] += grad2[w*total + total - outputs + i]

def mdl():
	null_arrs(var, locd, grad, meand, locd2, dlocd, d2var, grad2, meand2)
	set_inp()
	for param in params:
		dot1d_use(param)#w, var, locd, grad, meand, locd2, dlocd, d2var, grad2, meand2, param)

#S = L(want, get) ~ l
def score():
	mdl()
	return L()

#ddS/dxdy = (fxy - fx - fy + f)/1e-10 ~ n * n * 4*l
def calculer_1e10_grad2():
	_grad2_1e10 = []
	for i in range(len(w)):
		_grad2_1e10 += [[]]
		for j in range(len(w)):
			w[i] += 1e-5; w[j] += 1e-5
			fxy = score()

			w[j] -= 1e-5
			fx = score()
			
			w[i] -= 1e-5
			f = score()
			
			w[j] += 1e-5
			fy = score()
			w[j] -= 1e-5

			dxdy = (fxy-fx-fy+f)/1e-10
			_grad2_1e10[-1] += [dxdy]

	return _grad2_1e10

#dS/dx = forward_backward() ~ 2*l
def forward_backward():
	null_arrs(var, locd, grad, meand, locd2, dlocd, d2var, grad2, meand2)
	set_inp()
	for param in params:
		dot1d_grad_forward(param)
	dL()
	for param in params[::-1]:
		dot1d_grad_backward(param)
	return meand

#ddS/dxdy = (fb+ - fb)/1e-5 ~ n * 2 * l
def calculer_forward_backward_1e5_grad2():
	_grad2_1e5 = []
	for i in range(len(w)):
		w[i] += 1e-5
		fx = copy(forward_backward())

		w[i] -= 1e-5
		f = copy(forward_backward())

		_grad2_1e5 += [[]]
		for j in range(len(w)):
			_grad2_1e5[-1] += [(fx[j]-f[j])/1e-5]
	return [[_grad2_1e5[j][i] for j in range(weights)] for i in range(weights)]

#ddS/dxdy = f2_b2_bb2_bf2() ~ 4*l (ou plutot (1 + 1.2 + 2 + 1.2)*l = 5*l)
def forward2_backward2_backwardofbackward2_backwardofforward2():
	null_arrs(var, locd, grad, meand, locd2, dlocd, d2var, grad2, meand2)
	set_inp()
	for param in params:
		dot1d_grad2_f2(param)
	
	dL()
	for param in params[::-1]:
		dot1d_grad2_b2(param)

	#Que la diagonale
	for w in range(weights):
		Dmeand[w*weights + w] = 1

	for param in params:
		dot1d_grad2_bb2(param)
	
	ddL()
	
	for param in params[::-1]:
		dot1d_grad2_bf2(param)

	deriv_set_inp()
	
	#dddL()
	return meand2

if __name__ == "__main__":
	s0 = score()
	print(f"Score initiale de {s0}")
	start_w = copy(w)
	
	#SGD
	t = time()
	for echope in range(30):
		grad = forward_backward()
		for i in range(weights):
			w[i] -= grad[i] * 0.1
		
	print(f"SGD a pris {time()-t} pour arriver a {score()}")
	w = start_w
	t = time()
	
	for _ in range(2):
		meand2 = forward2_backward2_backwardofbackward2_backwardofforward2()
		l = sum(meand2)/weights
		print(1/l)
		for i in range(weights):
			w[i] -= grad[i]/l
	print(f"Hessienne a pris {time()-t} pour arriver a {score()}")

'''
if __name__ == "__main__":
	print(" === Start === ")

	#args = (data, w, var, locd, grad, meand, locd2, dlocd, d2var, grad2, meand2, params)

	#start = time()
	#mat2d = calculer_1e10_grad2()
	#end = time()
	#print_matrice(mat2d)
	#print(end - start)

	print("====================================")

	start = time()
	mat2d = calculer_forward_backward_1e5_grad2()
	end = time()
	#print_matrice(mat2d)
	print(end - start)
	t0 = end-start

	print("====================================")

	#
	#	Finnalement ca prend autant de temps car de tout facon c'est soit (for_back)*n ou (n)back_for_back
	#	car il y a un for dw in range(weights) dans tous les kernel du calcule a 4 etapes
	#

	start = time()
	mat2d = forward2_backward2_backwardofbackward2_backwardofforward2()
	end = time()
	mat2d = [[mat2d[i*weights + j] for j in range(weights)] for i in range(weights)]
	#print_matrice(mat2d)
	print(end - start)

	print("k = ", t0/(end-start))

	#print_state()'''