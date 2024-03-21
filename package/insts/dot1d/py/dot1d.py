from math import exp, tanh

from package.insts.activations import activate, localderiv, local2deriv

from kernel.py.inst import Inst
from package.insts.build_from_required import BuildFromRequired
from kernel.py.etc import pseudo_randomf

from random import random, randint, seed

class DOT1D(BuildFromRequired):

	name = "DOT1D"

	ALLOW_DF = True
	ALLOW_DDF = True

	params_names = ['Ax','Yx', 'activ', 'istart','ystart','wstart','lstart','l2start']

	################################ Mdl Kerd Funcs ##########################################

	def check_params(self):
		#drate == drop rate
		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params
		assert activ < len(activate) and Ax > 0 and Yx > 0 and all(i>=0 and int(i)==i for i in self.params)

	def check_model(self, insts_ids:[int], params:[[int]], this_inst_pos:int):
		#	Cette instruction peut etre inserer n'importe ou, pas de conditions spéciale
		#	Certaines instructions ont besoin d'etre inter-liés pour se paramettrer
		#check_model est juste un fonction qui va voire si cette instruction est coherente avec les autres (ex : des insts qui communiques)
		pass

	################################## Use ####################################

	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):

		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params

		for y in range(Yx):
			_sum = 0
			
			for i in range(Ax):
				_sum += var[l*total + istart + i] * w[wstart + i*Yx + y]
			_sum += w[wstart + Ax*Yx + y]

			var[l*total + ystart + y] = activate[activ](_sum)

	############################ F & dF ############################
			
	def forward(self, start_seed:int,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int,
		w:[float], var:[float], locd:[float]):

		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params

		for y in range(Yx):
			_sum = 0

			for i in range(Ax):
				
				Apos = sets*total*line + _set*total + istart + i
				
				#__seed = Apos * (start_seed+1)
				#if pseudo_randomf(__seed) >= drate:
				_sum += var[Apos] * w[ws*_set + wstart + i*Yx + y]
				
			_sum += w[ws*_set + wstart + Ax*Yx + y]

			locd[sets*line*locds + _set*locds + lstart + y] = localderiv[activ](_sum)
			var[sets*total*line + _set*total + ystart + y] = activate[activ](_sum)
			
	def backward(self, start_seed:int,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int,
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		
		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params 

		for y in range(Yx):
			dlds = locd[sets*line*locds + _set*locds + lstart + y] * grad[sets*total*line + _set*total + ystart + y]

			meand[ws*_set + wstart + Ax*Yx + y] += dlds

			for i in range(Ax):
				vpos = sets*total*line + _set*total + istart + i

				#if pseudo_randomf(vpos*(start_seed+1)) >= drate:
				wpos = ws*_set + wstart + i*Yx + y
				vpos = sets*total*line + _set*total + istart + i
					
				grad[vpos] += dlds * w[wpos]
				meand[wpos] += dlds * var[vpos]

	################################ ddF #####################################

	def forward2(self, start_seed:int,
		sets:int, vsize:int, wsize:int, lsize:int, l2size:int, _set:int, time:int,
		weight:[float], var:[float],
		locd:[float], grad:[float], meand:[float],
		locd2:[float],
		dd_weight:[float], dd_var:[float],
		dd_locd:[float], dd_grad:[float], dd_meand:[float]):

		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params

		for y in range(Yx):
			s = weight[_set*wsize + wstart + Ax*Yx + y]
			for k in range(Ax):
				s += var[time*sets*vsize + _set*vsize + istart + k] * weight[_set*wsize + wstart + k*Yx + y]
			var[time*sets*vsize + _set*vsize + ystart + y] = activate[activ](s)
			locd[time*sets*lsize + _set*lsize + lstart + y] = localderiv[activ](s)
			locd2[time*sets*l2size + _set*l2size + l2start + y] = local2deriv[activ](s)

	def backward2(self, start_seed:int,
		sets:int, vsize:int, wsize:int, lsize:int, l2size:int, _set:int, time:int,
		weight:[float], var:[float],
		locd:[float], grad:[float], meand:[float],
		dd_weight:[float], dd_var:[float],
		dd_locd:[float], dd_grad:[float], dd_meand:[float]):

		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params

		for y in range(Yx):
			dlds = grad[time*sets*vsize + _set*vsize + ystart + y] * locd[time*sets*lsize + _set*lsize + lstart + y]

			meand[_set*wsize + wstart + Ax*Yx + y] += dlds

			for k in range(Ax):
				#s += var[istart + k] * w[wstart + k*Yx + y]
				grad[time*sets*vsize + _set*vsize + istart + k] += dlds * weight[_set*wsize + wstart + k*Yx + y]
				meand[_set*wsize + wstart + k*Yx + y] += dlds * var[time*sets*vsize + _set*vsize + istart + k]

	def backward_of_backward2(self, start_seed:int,
		dw:int,	#Matrice hessienne c'est d(dS/dw)/d[] on fait ligne par ligne (plus opti)
		sets:int, vsize:int, wsize:int, lsize:int, l2size:int, _set:int, time:int,
		weight:[float], var:[float],
		locd:[float], grad:[float], meand:[float],
		locd2:[float],
		dd_weight:[float], dd_var:[float],
		dd_locd:[float], dd_grad:[float], dd_meand:[float]):

		#[dd_w]<sets*weight*weights> en python il n'y a pas de pointeurs, donc il faut dw*sets*weights + _set*weights + w
		#[dd_var, dd_locd, dd_grad, dd_meand]<time*sets> dS/dw

		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params

		for y in range(Yx):
			dlds = grad[time*sets*vsize + _set*vsize + ystart + y] * locd[time*sets*lsize + _set*lsize + lstart + y]

			#meand[_set*wsize + wstart + Ax*Yx + y] += dlds

			Ddlds = 0

			for k in range(Ax):
				#grad[istart + k] += dlds * w[wstart + k*Yx + y]
				_grad2 = dd_grad[time*sets*vsize + _set*vsize + istart + k]
				if _grad2 != 0:
					Ddlds += _grad2 * weight[_set*wsize + wstart + k*Yx + y]
					dd_weight[dw*sets*weights + _set*wsize + wstart + k*Yx + y] += _grad2 * dlds

				#meand[wstart + k*Yx + y] += dlds * var[istart + k]
				_dd_meand = dd_meand[_set*wsize + wstart + k*Yx + y]
				if _dd_meand != 0:
					Ddlds += var[time*sets*vsize + _set*vsize + istart + k] * _dd_meand
					dd_var[time*sets*vsize + _set*vsize +  + istart + k] += dlds * _dd_meand

			Ddlds += dd_meand[_set*wsize + wstart + Ax*Yx + y]
			
			if Ddlds != 0:
				dd_grad[time*sets*vsize + _set*vsize + ystart + y] += locd[time*sets*lsize + _set*lsize + lstart + y] * Ddlds
				dd_locd[time*sets*lsize + _set*lsize + lstart + y] += grad[time*sets*vsize + _set*vsize + ystart + y] * Ddlds

	def backward_of_forward2(self, start_seed:int,
		dw:int,	#Matrice hessienne c'est d(dS/dw)/d[] on fait ligne par ligne (plus opti)
		sets:int, vsize:int, wsize:int, lsize:int, l2size:int, _set:int, time:int,
		weight:[float], var:[float],
		locd:[float], grad:[float], meand:[float],
		locd2:[float],
		dd_weight:[float], dd_var:[float],
		dd_locd:[float], dd_grad:[float], dd_meand:[float]):

		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params

		for y in range(Yx):
			#s = w[wstart + Ax*Yx + y]

			#for k in range(Ax):
			#	s += var[istart + k] * w[wstart + k*Yx + y]
				
			#var[ystart + y] = tanh(s)
			#locd[lstart + y] = 1 - tanh(s)**2
			#locd2[lstart + y] = -2*tanh(s)*(1 - tanh(s)**2)

			ds = 0
			ds += locd2[time*sets*l2size + _set*l2size + l2start + y] * dd_locd[time*sets*lsize + _set*lsize + lstart + y]
			ds += locd[time*sets*lsize + _set*lsize + lstart + y] * dd_var[time*sets*vsize + _set*vsize + ystart + y]
			if ds != 0:
				for k in range(Ax):
					#s += var[istart + k] * w[wstart + k*Yx + y]
					dd_var[time*sets*vsize + _set*vsize + istart + k] += ds * weight[_set*wsize + wstart + k*Yx + y]
					dd_weight[dw*sets*wsize + _set*wsize + wstart + k*Yx + y] += ds * var[time*sets*vsize + _set*vsize + istart + k]

			dd_weight[dw*sets*wsize + _set*wsize + wstart + Ax*Yx + y] += ds

	####################### Spetial functions for applications ##########################

	#### Build Stack Model  (Applications : "stack_model.py", )

	def return_iywll2_start(self):
		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params
		return istart, ystart, wstart, lstart, l2start

	def relativ_ystart(self):
		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params
		return 0

	def buildstackmodel_vars(self):
		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params
		return Yx

	def buildstackmodel_weights(self):
		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params
		return Ax*Yx + Yx

	def buildstackmodel_locds(self):
		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params
		return Yx

	def buildstackmodel_locd2s(self):
		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params
		return Yx

	#### Labels Stack Model  (Applications : "stack_model.py", )

	def labelstackmodel_vars(self, _id, stack_start):
		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params
		return [(f'{_id}.Y [dot1d]',stack_start)]

	def labelstackmodel_weights(self, _id, stack_start):
		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params
		return [(f'{_id}.W [dot1d]',stack_start), (f'{_id}.B [dot1d]',stack_start + Ax*Yx)]

	def labelstackmodel_locds(self, _id, stack_start):
		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params
		return [(f'{_id}.df [dot1d]',stack_start)]

	def labelstackmodel_locd2s(self, _id, stack_start):
		Ax, Yx, activ, istart, ystart, wstart, lstart, l2start = self.params
		return [(f'{_id}.ddf [dot1d]',stack_start)]

	### Setput Params Stack Model  (Applications : "stack_model.py", )

	requiredforsetupparams = "Ax", "Yx", "activ"		#build vars,weights and locd have to ask only for thoses params

	requiredposition = 1,1,1,0,0,0,0,0 #1,1,1 == Ax,Yx,activ

	params_defaults = {
		'activ' : 0,
	}

	def setupparamsstackmodel(self, istart, ystart, wstart, lstart, l2start, required):
		Ax, Yx, activ = required
		return Ax, Yx, activ, istart, ystart, wstart, lstart, l2start

	### Check Params Input output

	def need_inputs(self, required):
		Ax, Yx, activ = required
		return Ax

	def check_input_output(self, last_vars, required):	#verifier que le vars du l'instruction precedente est de la meme taille 
		Ax, Yx, activ = required				#que le input pour cet instruction
		assert last_vars == self.need_inputs(required)

	####################### Agnostic Network ##########################

	class agnostic_random_network1d_tensor4d:

		#Par exemple les Ax,Ay,Az,An doivent etre indépandantes les unes des autres
		compatible = True

		@staticmethod
		def generate_next_tensor_and_required(lastTens, minsA, maxsA, required_rnd_seed, libre_rnd_seed):
			Ax, Ay, Az, An = lastTens
			minx, miny, minz, minn = minsA
			maxx, maxy, maxz, maxn = maxsA

			seed(required_rnd_seed)
			nextTens = [
				randint(minx, maxx),
				1,
				1,
				1
			]

			seed(libre_rnd_seed)
			required = {
				'Ax' : Ax*Ay*Az*An,
				'Yx' : nextTens[0],
				'activ' : randint(0, len(activate)-1)
			}

			return nextTens, required

		#@staticmethod pour aucune confusion
		@staticmethod
		def is_linkable(inp_tens, out_tens):
			#	Cette fonction est surtout utile pour lier la derniere instruction avec l'output (ou determiner les instructions qui sont liables)
			Ax, Ay, Az, An = inp_tens
			Bx, By, Bz, Bn = out_tens

			#Peut importe les situations dot1d lie un vecteur a un autre
			#Les elements des tenseurs sont juste multiplié entre eux
			#Par contre dotconvl1d par exemple ne peut pas lier 5 a 12 par exemple, car plus grand
			return True
	
		required = {
			'Ax' : 'relatif', 	#relatif aux tenseur input et output
			'Yx' : 'relatif',
			'activ' : 'libre',
		}

		@staticmethod
		def build_required_relatif(inp_tens, out_tens):
			inpAx, inpAy, inpAz, inpAn = inp_tens
			outAx, outAy, outAz, outAn = out_tens
			
			return {
				'Ax' : inpAx*inpAy*inpAz*inpAn,
				'Yx' : outAx*outAy*outAz*outAn
			}

		@staticmethod
		def build_required_libre(libre_rnd_seed):
			seed(libre_rnd_seed)
			return {
				'activ' : randint(0, len(activate)-1),
			}