#! /usr/bin/python3

from math import exp, tanh, pi
import matplotlib.pyplot as plt
from random import randint, random, seed
from os import system

seed(1000)

def auto_gen(x, w, mdl, N):
	mdl.restart()
	y = [x]
	for _ in range(N):
		y += [mdl(y[-1], w)]
	return [i for b in y[1:] for i in b]

#	Pour un meme input, comment diffrentes listes de poids evolent
def variabilite_modelisation(Is:'differentes entree', Vs:'differentes version de poids', mdls):
	inputs, outputs = len(data[0][0][0]), len(data[0][0][1])
	assert inputs == outputs
	N = inputs
	#
	fig, ax = plt.subplots(Is, len(mdls))
	for bid in range(Is):
		###
		x = [random()-.5 for _ in range(inputs)]
		###
		for p in range(len(mdls)):
			for _ in range(Vs):
				#mdls[p].restart()
				#pred = mdls[p](x,[3*(random()-.5) for _ in range(len(mdls[p].w))])
				pred = auto_gen(x, [2*(random()-.5) for _ in range(len(mdls[p].w))], mdls[p], 3)
				eval(f"ax{'[bid]' if Is!=1 else ''}{'[p]' if len(mdls)!=1 else ''}").plot(pred, 'r')

			eval(f"ax{'[bid]' if Is!=1 else ''}{'[p]' if len(mdls)!=1 else ''}").set_title(str(MDLS[p]).split('.')[-1][:-2])
	plt.show()

#	Pour une meme liste de poids comment les entree font varier le resultat
def variabilite_resultat(Is:'differentes entree', Vs:'differentes version de poids', mdls):
	inputs, outputs = len(data[0][0][0]), len(data[0][0][1])
	assert inputs == outputs
	N = inputs
	#
	fig, ax = plt.subplots(Is, len(mdls))
	for bid in range(Is):
		###
		ws = [[2*(random()-.5) for _ in range(len(mdls[p].w))] for p in range(len(mdls))]
		###
		for p in range(len(mdls)):
			for _ in range(Vs):
				#mdls[p].restart()
				#pred = mdls[p]([(random()-.5) for _ in range(inputs)], ws[p])
				pred = auto_gen([(random()-.5) for _ in range(inputs)], ws[p], mdls[p], 2)
				eval(f"ax{'[bid]' if Is!=1 else ''}{'[p]' if len(mdls)!=1 else ''}").plot(pred, 'r')

			eval(f"ax{'[bid]' if Is!=1 else ''}{'[p]' if len(mdls)!=1 else ''}").set_title(str(MDLS[p]).split('.')[-1][:-2])
	plt.show()

########################################################################################

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

########################################################################################""

def plot_batch(batch, w):
	pred = f(batch, w)
	pred = [([(i+1)*inputs-1]+[j+(i+1)*inputs for j in range(inputs)], [data[i][0][-1]]+_pred) for i,_pred in enumerate(pred)]
	plt.plot([i for i in range(len(serie))], serie, label="donnee")
	for inp,out in pred:
		plt.plot(inp, out, 'r')
	plt.show()

def plot_batch_versions_2D(data, mdls, ws):
	global mdl
	inputs, outputs = len(data[0][0][0]), len(data[0][0][1])
	assert inputs == outputs
	N = inputs
	#
	fig, ax = plt.subplots(len(data), len(mdls))
	for bid, batch in enumerate(data):
		###
		serie = cpy(batch[0][0])
		for inp,out in batch: serie += out
		###
		for p in range(len(mdls)):
			mdl = mdls[p]
			mdls[p].restart()
			pred = f(batch,ws[p])
			pred = [([(i+1)*N-1]+[(i+1)*N+j for j in range(N)],  [batch[i][0][-1]]+get) for i,get in enumerate(pred)]
				
			eval(f"ax{'[bid]' if len(data)!=1 else ''}{'[p]' if len(mdls)!=1 else ''}").plot([i for i in range(len(serie))], serie, label="donnee")
				
			for numeros,_pred in pred:
				eval(f"ax{'[bid]' if len(data)!=1 else ''}{'[p]' if len(mdls)!=1 else ''}").plot(numeros, _pred, 'r')

			eval(f"ax{'[bid]' if len(data)!=1 else ''}{'[p]' if len(mdls)!=1 else ''}").set_title(str(MDLS[p]).split('.')[-1][:-2])
	plt.show()

########################################################################################""

#	Batch learning (1 Batch en tout, si on veut d'autres batchs on recree data d'une autre maniere)

aj = lambda lst,i: lst[:i]+[lst[i]+1e-5]+lst[i+1:]
flat = lambda arr: [i for j in arr for i in j]
cpy = lambda arr: [x for x in arr]

def f(batch, w):
	mdl.restart()
	return [mdl(x,w) for x,y in batch]
S = lambda batch, w: sum( sum( (get-want)**2/2  for get,want in zip(fs,out)) for fs,(inp,out) in zip(f(batch,w), batch)) / ( len(batch[0]) * len(batch[0][0]) )
dS = lambda batch, w,i: (S(batch, aj(w,i))-S(batch, w))*1e5

def grad(batch, w):
	return [dS(batch, w, i=i) for i in range(len(w))]

def tableau(batch, w):
	return [(dS(batch, aj(w,dw),i)-dS(batch,w,i))*1e5 for dw in range(len(w)) for i in range(len(w))]

def tableau_if(batch, w, _grad):
	return [((dS(batch, aj(w,dw),i)-dS(batch,w,i))*1e5 if _grad[dw]!=0 else 0) for dw in range(len(w)) for i in range(len(w))]

########################################################################################""

def grad_descente(batch, w, alpha=0.1):
	_grad = grad(batch, w)
	for i in range(len(w)): w[i] -= alpha*_grad[i]
	return w

def hess_descente(batch, w):
	_grad = grad(batch, w)
	_tableau = tableau_if(batch,w, _grad)

	inv = inverser(_tableau)

	if inv == False:
		print("Matrice Non Inversible")
		return w
		#exit()

	for i in range(len(w)):
		w[i] -= sum(_grad[j]*inv[i*len(w) + j] for j in range(len(w)))/len(w)
	
	return w

def taton(batch, w):
	_dw = randint(0, len(w)-1)
	
	_1 = S(batch, w)
	w[_dw] += 0.05
	_2 = S(batch, w)
	w[_dw] -= 2*0.05
	_3 = S(batch, w)
	_m = min(_1,_2,_3)
	if _1 == _m: w[_dw] += 2*0.05
	if _2 == _m: w[_dw] += 0.05

	return w

def rnd(batch, w):
	_w = [3*(random()-.5) for _ in range(len(w))]

	if S(batch, _w) < S(batch, w):
		for i in range(len(w)):
			w[i] = _w[i]

	return w

########################################################################################""

def cree_serie(inputs, outputs, lignes=5):
	assert outputs == inputs	#pour l'instant

	N = inputs

	s = 0
	h = 0
	bloques = []
	for i in range(lignes+1):
		bloques += [[]]
		for _ in range(N):
			s += random()-0.5 + h
			#h += 0.5*(random()-0.5)
			bloques[-1] += [s]

	return [
		(bloques[i], bloques[i+1]) for i in range(lignes)
	]

skips = ("mem1d", "filtre_dot1d", "mem_dot1d")
MDLS = []
from os import listdir
for i in listdir("tests"):
	if '.py' in i:
		i = i.replace('.py', '')
		print(i, end=("\n" if not i in skips else " [SKIPP]\n"))
		if not i in skips:
			exec(f"from tests.{i} import {i.upper()}")
			exec(f"MDLS += [{i.upper()}]")

########################################################################################

def iterations(I, fonction, ECHO, *args, **kargs):
	for it in range(I):
		acc = "#" * it + " "*(I-it)
		if ECHO: system(f'clear;printf "|{acc}|"')

		batch = data[randint(0, BATCHS-1)]

		fonction(batch, *args, **kargs)

	if ECHO: print("")

if __name__ == "__main__":
	N = 10
	BATCHS = 1
	LIGNES = 10

	data = [
		cree_serie(N, N, lignes=LIGNES) for _ in range(BATCHS)
	]

	mdls = []
	ws = []

	for MDL in MDLS:
		mdl = MDL(N, N)
		w = [random()-.5 for _ in range(len(mdl.w))]

		alpha = 0.1# / ( sum(tableau(data[0], w)) / len(w)*len(w))
		#print(alpha)

		#iterations(50, grad_descente, False, w, alpha=alpha)
		#iterations(10, hess_descente, True, w)
		#iterations(1000, taton, False, w)
		#iterations(10000, rnd, False, w)

		mdls += [mdl]
		ws += [w]

	#plot_batch_versions_2D(data, mdls, ws)

	#variabilite_modelisation(5, 10, mdls)
	variabilite_resultat(5, 10, mdls)