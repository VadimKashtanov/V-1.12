'''

	Ceci ajoute n'importe quel instruction avec

$python3 add_inst.py dotgaussfiltre3d th11 Ax,Ay,Az,Bx,istart,ystart,wstart,lstart Ax,Ay,Az,Bx

Ici le nom sera : dotgaussfiltre3d
le mode : th11
les parametres : Ax,Ay,Az,Bx,istart,ystart,wstart,lstart
les parametres requis : Ax,Ay,Az,Bx    (relatif a mon package)
'''

build_python = lambda name, params, req: f"""from kernel.py.inst import Inst
from package.insts.build_from_required import BuildFromRequired

from random import randint, seed

class {name.upper()}(BuildFromRequired):

	name = "{name.upper()}"

	params_names = [{''.join("'" + param + "', " for param in params)}]

	################################ Kernel Functions ##########################################
	def check(self):
		{', '.join(params)} = self.params
		assert all(i>=0 and int(i)==i for i in self.params)

	def check_model(self, insts_ids:[int], params:[[int]], this_inst_pos:int):
		#	Cette instruction peut etre inserer n'importe ou, pas de conditions spéciale
		#	Certaines instructions ont besoin d'etre inter-liés pour se paramettrer
		pass

	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):

		{', '.join(params)} = self.params

		for y in range(Yx):
			_sum = 0
			
			for i in range(Ax):
				_sum += var[l*total + istart + i] * w[wstart + i*Yx + y]
			_sum += w[wstart + Ax*Yx + y]

			var[l*total + ystart + y] = activate[activ](_sum)

	def forward(self, start_seed:int,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int,
		w:[float], var:[float], locd:[float]):
		
		{', '.join(params)} = self.params

		for y in range(Yx):
			_sum = 0
			for i in range(Ax):
				_sum += var[sets*total*line + _set*total + istart + i] * w[ws*_set + wstart + i*Yx + y]
			_sum += w[ws*_set + wstart + Ax*Yx + y]

			locd[sets*line*locds + _set*locds + lstart + y] = localderiv[activ](_sum)
			var[sets*total*line + _set*total + ystart + y] = activate[activ](_sum)

	def backward(self, start_seed:int,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int,
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		
		{', '.join(params)} = self.params

		for y in range(Yx):
			dlds = locd[sets*line*locds + _set*locds + lstart + y] * grad[sets*total*line + _set*total + ystart + y]

			meand[ws*_set + wstart + Ax*Yx + y] += dlds

			for i in range(Ax):
				wpos = ws*_set + wstart + i*Yx + y
				vpos = sets*total*line + _set*total + istart + i
				grad[vpos] += dlds * w[wpos]
				meand[wpos] += dlds * var[vpos]

	####################### Spetial functions for applications ##########################

	#### Build Stack Model  (Applications : "stack_model.py", )

	def return_iywl_start(self):
		{', '.join(params)} = self.params
		return istart, ystart, wstart, lstart #Remplace missing by `None`

	def relativ_ystart(self):	#The instruction can store N vars, but only 10% of it is the output. Exemple LSTM have to save `e` and `h` but only `h` is output
		{', '.join(params)} = self.params
		return 0

	def buildstackmodel_vars(self):
		{', '.join(params)} = self.params
		return 0

	def buildstackmodel_weights(self):
		{', '.join(params)} = self.params
		return 0

	def buildstackmodel_locds(self):
		{', '.join(params)} = self.params
		return 0

	#### Labels Stack Model  (Applications : "stack_model.py", )

	def labelstackmodel_vars(self, _id, stack_start):
		{', '.join(params)} = self.params
		return [(f'{{_id}}.Y [{name}]',stack_start)]

	def labelstackmodel_weights(self, _id, stack_start):
		{', '.join(params)} = self.params
		return [(f'{{_id}}.W [{name}]',stack_start)]

	def labelstackmodel_locds(self, _id, stack_start):
		{', '.join(params)} = self.params
		return [(f'{{_id}}.Y [{name}]',stack_start)]

	### Setput Params Stack Model  (Applications : "stack_model.py", )

	requiredforsetupparams = {''.join('"' + r + '", ' for r in req)}

	requiredposition = {', '.join(str(int(param in req)) for param in params)}

	params_defaults = {{
		#Que les parametres libres comme <drate> ou <activ>
	}}

	def setupparamsstackmodel(self, istart, ystart, wstart, lstart, required):
		{', '.join(req)} = required
		return {', '.join(params)}

	### Check Params Input output

	def need_inputs(self, required):
		{', '.join(req)} = required
		return 0

	def check_input_output(self, last_vars, required):
		{', '.join(req)} = required
		assert last_vars == self.need_inputs(required)

	####################### Agnostic Network ##########################

	class agnostic_random_network1d_tensor4d:

		#Par exemple les Ax,Ay,Az,An doivent etre indépandantes les unes des autres
		compatible = False
		#Il faut *absolument* que l'ont puisse lier 

		@staticmethod
		def generate_next_tensor_and_required(lastTens, minsA, maxsA, required_rnd_seed, libre_rnd_seed):
			Ax, Ay, Az, An = lastTens
			minx, miny, minz, minn = minsA
			maxx, maxy, maxz, maxn = maxsA

			seed(required_rnd_seed)
			nextTens = [	#Je pourrais random partout
				randint(minx, maxx),
				1,
				1,
				1
			]

			seed(libre_rnd_seed)
			required = {{
				'Ax' : Ax*Ay*Az*An,
				'Yx' : nextTens[0],
				'activ' : randint(0, len(activate)-1),
				'drate' : 0 #randint(0, 100)
			}}

			return nextTens, required

		@staticmethod
		def is_linkable(inp_tens, out_tens):
			Ax, Ay, Az, An = inp_tens
			Bx, By, Bz, Bn = out_tens

			#Peut importe les situations dot1d lie un vecteur a un autre
			#Les elements des tenseurs sont juste multiplié entre eux
			#Par contre dotconvl1d par exemple ne peut pas lier 5 a 12 par exemple, car plus grand
			return True
	
		required = {{
			'Ax' : 'relatif', 	#relatif aux tenseur input et output
			'Yx' : 'relatif',
			'activ' : 'libre',
			'drate' : 'libre'
		}}

		@staticmethod
		def build_required_relatif(inp_tens, out_tens, required_rnd_seed):
			inpAx, inpAy, inpAz, inpAn = inp_tens
			outAx, outAy, outAz, outAn = out_tens

			seed(required_rnd_seed)
			
			return {{
				'Ax' : inpAx*inpAy*inpAz*inpAn,
				'Yx' : outAx*outAy*outAz*outAn
			}}

		@staticmethod
		def build_required_libre(libre_rnd_seed):
			seed(libre_rnd_seed)
			return {{
				'activ' : randint(0, len(activate)-1),
				'drate' : 0,#randint(0, 100)
			}}
"""

build_python_test_package = lambda name, params, req: f"""from package.insts.{name}.py.{name} import {name.upper()}
from kernel.py.mdl import Mdl
from package.insts.fast_model import Fast_1Layer_FeedForward_Mdl
from package.scores.meansquared.py.meansquared import MEANSQUARED
from kernel.py.test_package import Test_MDL
from random import random, seed

seed(0)

class TEST_MDL_{name.upper()}(Test_MDL):
	score_algo = 0
	score_class = MEANSQUARED
	score_consts = []
	
	mdl = Fast_1Layer_FeedForward_Mdl(
		inst={name.upper()},
		required=[{', '.join(req)}]
	).build_weights()

	### compare .mdl() and .forward()
	###		ex: drop_rate=0
	mdl_mdl_forward_compare = Fast_1Layer_FeedForward_Mdl(
		inst:={name.upper()},
		required:=[{', '.join(req)}]
	).build_weights()

	lines = 2
	sets = 4

	scores_args = [
		[],	#mean squared
		[]	#cross entropy
	]

	optis_args = [
		[],	#sgd
		[],	#momentum
		[],	#rmsprop
		[],	#adam
	]
"""

build_python_paper_results = lambda name, params, req: f"""from package.insts.{name.lower()}.py.{name.lower()} import {name.upper()}
from package.insts.fast_model import Fast_1Layer_FeedForward_Mdl

class PAPER_RESULT_{name.upper()}_0:
	mdl = Fast_1Layer_FeedForward_Mdl(
		inst:={name.upper()},
		required:=[{''.join([r+':=1, ' for r in req])}]
	)

	lines = 2

	#Il faut utiliser f_de_x.py pour qu'il ecrive le resultat d'une instruction a partire de weights et inputs
	weights = []

	#Donc inputs+vars  (et vars = non_output_var + outputs)
	variables = [
		#Ligne == 0
		0,

		#Ligne == 1
		0
	]

PAPER_RESULT_{name.upper()} = [PAPER_RESULT_{name.upper()}_0]
"""

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

build_head = lambda name, mods, params, req: f"""#pragma once

#include "kernel/head/train.cuh"

#include "{name}_th11.cuh"

#include "package/insts/activation.cuh"

//=========================== Sizes ===============================

/*
	Params = [{', '.join(params)}]

*/

//=========================== Sizes ===============================

/*
	inputs = 
	vars = 
	weights = 
	locds = 
*/

void {name}_check(uint * param);

//======================= Cpu_t forward ===========================

void {name}_cpu(Cpu_t * cpu, uint inst, uint time);

//======================= Use_t Forward ===========================

""" + "\n".join(f"void {name}_use_call_mode_{mod}(Use_t * use, uint inst, uint time);" for mod in mods) + f"""

void {name}_use(Use_t * use, uint inst, uint time);

//======================== Train_t =======================

//-------------------------- forward ---------------------

""" + "\n".join(f"void {name}_forward_call_mode_{mod}(Train_t * train, uint inst, uint time, uint start_seed);" for mod in mods) + f"""

void {name}_forward(Train_t * train, uint inst, uint time, uint start_seed);

//-------------------------- backward ---------------------

""" + "\n".join(f"void {name}_backward_call_mode_{mod}(Train_t * train, uint inst, uint time, uint start_seed);" for mod in mods) + f"""

void {name}_backward(Train_t * train, uint inst, uint time, uint start_seed);

"""

build_inst_cu = lambda name, mods, params, req: f"""#include "package/insts/{name}/head/{name}.cuh"

void {name}_check(uint * param) {{
	if (param[0] == 0) raise(SIGINT);
	if (param[1] == 0) raise(SIGINT);
}};

void {name}_cpu(Cpu_t * cpu, uint inst, uint time) {{
	Mdl_t * mdl = cpu->mdl;
	uint total = mdl->total;

	uint * param = mdl->param[inst];

""" + "".join(f"\tuint {param}=param[{i}];\n" for i,param in enumerate(params)) + f"""

	float * var = cpu->var;
	float * weight = mdl->weight;

	var[time*total + ystart + i];
	var[time*total + istart + i];
}};

void {name}_use(Use_t * use, uint inst, uint time) {{
""" + "\n\t".join(f"\t{name}_use_call_mode_{mod}(use, inst, time);" for mod in mods) + f"""
}};

void {name}_forward(Train_t * train, uint inst, uint time, uint start_seed) {{
""" + "\n\t".join(f"\t{name}_forward_call_mode_{mod}(train, inst, time, start_seed);" for mod in mods) + f"""
}};

void {name}_backward(Train_t * train, uint inst, uint time, uint start_seed) {{
""" + "\n".join(f"\t{name}_backward_call_mode_{mod}(train, inst, time, start_seed);" for mod in mods) + f"""
}};

"""

build_call_mod = lambda name, mods, params, req: f"""#include "package/insts/{name}/head/{name}.cuh"

void {name}_use_call_mode_{mod}(Use_t * use, uint inst, uint time) {{
	Mdl_t * mdl = use->mdl;

	uint * params = mdl->param[inst];

""" + ''.join(f"\t\t {param}=params[{i}],\t\t\\\n" for i,param in enumerate(params)) + f"""

	{name}_use_{mod}<<<dim3(KERN_DIV(Bx,8), KERN_DIV(Ay,8), KERN_DIV(Az,8)),dim3(8,8,8)>>>(
		{''.join(map(lambda x:x+", ",req))}
		time,
		mdl->total, mdl->weights,
		istart, ystart, wstart,
		use->var_d, use->weight_d);
	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());
}};

//======================== Train_t =======================

//-------------------------- forward ---------------------

void {name}_forward_call_mode_{mod}(Train_t * train, uint inst, uint time, uint start_seed) {{
	Mdl_t * mdl = train->mdl;
	uint sets = train->sets;

	uint * params = mdl->param[inst];

""" + ''.join(f"\t\t {param}=params[{i}],\t\t\\\n" for i,param in enumerate(params)) + f"""

	{name}_forward_{mod}<<<dim3(KERN_DIV(Bx,8), KERN_DIV(Ay,8), KERN_DIV(Az,8)),dim3(8,8,8)>>>(
		{''.join(map(lambda x:x+", ",req))}
		time,
		train->mdl->total, train->mdl->weights, train->mdl->locds,
		istart, ystart, wstart, lstart,
		inst*start_seed, drop_rate,
		train->sets,
		train->_var, train->_weight, train->_locd);
	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());
}};

//-------------------------- backward ---------------------

void {name}_backward_call_mode_{mod}(Train_t * train, uint inst, uint time, uint start_seed) {{
	Mdl_t * mdl = train->mdl;
	uint sets = train->sets;

	uint * params = mdl->param[inst];

""" + ''.join(f"\t\t {param}=params[{i}],\t\t\\\n" for i,param in enumerate(params)) + f"""

	{name}_backward_{mod}<<<dim3(KERN_DIV(Bx,8), KERN_DIV(Ay,8), KERN_DIV(Az,8)),dim3(8,8,8)>>>(
		{''.join(map(lambda x:x+", ",req))}
		time,
		mdl->total, mdl->weights, mdl->locds,
		istart, ystart, wstart, lstart,
		inst*start_seed, drop_rate,
		train->sets,
		train->_var, train->_weight, train->_locd, train->_grad, train->_meand);
	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());
}}
"""

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

build_head_mod = lambda name, mod, params, req: f"""#pragma once

#include "kernel/head/train.cuh"

//	================== Use ==================

__global__
void {name}_use_{mod}(
	{''.join(map(lambda x:"uint "+x+", ",req))}
	uint time,
	uint total, uint wsize,
	uint istart,  uint ystart, uint wstart,
	float * var, float * weight);

//========================		Train_t	  =========================

//----------------------------- forward ---------------------------

__global__
void {name}_forward_{mod}(
	{''.join(map(lambda x:"uint "+x+", ",req))}				
	uint time,
	uint total, uint wsize, uint lsize,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint seed, float drop_rate,
	uint sets,
	float * var, float * weight, float * locd);

//----------------------------- backward ---------------------------

__global__
void {name}_backward_{mod}(
	{''.join(map(lambda x:"uint "+x+", ",req))}
	uint time,
	uint total, uint wsize, uint lsize,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint seed, float drop_rate,
	uint sets,
	float * var, float * weight, float * locd,
	float * grad, float * meand);
"""

build_use_mod = lambda name, mod, params, req: f"""#include "package/insts/{name}/head/{name}.cuh"

__global__
void {name}_use_{mod}(
	{''.join(map(lambda x:"uint "+x+", ",req))}
	uint time,
	uint total, uint wsize,
	uint istart, uint ystart, uint wstart,
	float * var, float * weight)
{{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	uint z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x < X && y < Y && z < Z) {{
		var[time*total + ystart + i];
	}}
}}

"""

build_forward_mod = lambda name, mod, params, req: f"""#include "package/insts/{name}/head/{name}.cuh"

__global__
void {name}_forward_{mod}(
	{''.join(map(lambda x:"uint "+x+", ",req))}
	uint time,
	uint total, uint wsize, uint lsize,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint seed, float drop_rate,
	uint sets,
	float * var, float * weight, float * locd)
{{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	uint z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x < X && y < Y && z < Z) {{
		var[time*sets*total + set*total + ystart + i];
	}}
}}
"""

build_backward_mod = lambda name, mod, params, req: f"""#include "package/insts/{name}/head/{name}.cuh"

__global__
void {name}_backward_{mod}(
	{''.join(map(lambda x:"uint "+x+", ",req))}
	uint time,
	uint total, uint wsize, uint lsize,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint seed, float drop_rate,
	uint sets,
	float * var, float * weight, float * locd,
	float * grad, float * meand)
{{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	uint z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x < X && y < Y && z < Z) {{
		var[time*sets*total + set*total + ystart + i];
	}}
}}
"""

def write_file(file, _str):
	with open(file, "w") as co:
		co.write(_str)

if __name__ == "__main__":
	#python3 add_inst.py time1dcopy th11 size,At,istart,ystart size,Ax
	
	from os import listdir, system
	from sys import argv

	#	Checking name [1]
	_object = "inst"
	_name = argv[1]
	assert not _name in listdir(f'package/insts')

	#	Build modes   [2]
	_modes = argv[2].split(',')
	assert all(_modes.count(mod) == 1 for mod in _modes)	#verifier qu'ils ne sont pas les memes
	assert not "mod" in _modes #pour etre sur qu'il n'y ai pas nom affin de copier coller correctement

	#	All args	  [3]
	params = argv[3].split(',')
	assert all(params.count(arg) == 1 for arg in params)	#verifier qu'ils ne sont pas les memes

	#	Build modes   [4]
	required = argv[4].split(',')
	assert all(required.count(req) == 1 for req in required)	#verifier qu'ils ne sont pas les memes

	### None-Required
	none_required = [param for param in params if not param in required]
	_params_without = [p for p in params]
	for np in none_required:
		del _params_without[_params_without.index(np)]

	assert all(_params_without[i] == required[i] for i in range(len(required)))	#	If params and required in same order

	#	Confirming
	if input(f"You want to build package/insts/{_name} with mods {_modes} ? (yes / no)").replace(' ','') == 'yes':
		dirs = [
			f"package/insts/{_name}",
			f"package/insts/{_name}/head",
			f"package/insts/{_name}/py",
			f"package/insts/{_name}/src/",
		] + [
			f"package/insts/{_name}/src/{mod}"
			for mod in _modes
		]

		for _dir in dirs:
			system(f"mkdir {_dir}")

		######################## General ####################################

		###		Files that are required for python
		system(f"touch package/insts/{_name}/__init__.py")
		write_file(f"package/insts/{_name}/py/{_name}.py", build_python(_name, params, required))
		write_file(f"package/insts/{_name}/py/test_mdl_{_name}.py", build_python_test_package(_name, params, required))
		write_file(f"package/insts/{_name}/py/paper_results_{_name}.py", build_python_paper_results(_name, params, required))

		###		General .cuh header
		write_file(f"package/insts/{_name}/head/{_name}.cuh", build_head(_name, _modes, params, required))
		write_file(f"package/insts/{_name}/src/{_name}.cu", build_inst_cu(_name, _modes, params, required))

		###################### C/Cuda Computing Modes #########################
		###	Building all Mods ###
		for mod in _modes:
			write_file(f"package/insts/{_name}/head/{_name}_{mod}.cuh", build_head_mod(_name, mod, params, required))
			write_file(f"package/insts/{_name}/src/{_name}_call_{mod}.cu", build_call_mod(_name, mod, params, required))

			################## /!\ /!\ Add _cal_mod /!\ /!\ ############################
			#	Do not rm -r dotgaussfiltre3d
			#	Add build_call in this add_insts.py
			#	Add dotgaussfiltre3d to use, forward, backward
			#	Add _call_th11 function and file to dotgaussfiltre3d
			#

			write_file(f"package/insts/{_name}/src/{mod}/{_name}_use_{mod}.cu", build_use_mod(_name, mod, params, required))
			write_file(f"package/insts/{_name}/src/{mod}/{_name}_forward_{mod}.cu", build_forward_mod(_name, mod, params, required))
			write_file(f"package/insts/{_name}/src/{mod}/{_name}_backward_{mod}.cu", build_backward_mod(_name, mod, params, required))
	else:
		print("Cancelling")