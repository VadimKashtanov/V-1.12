from kernel.py.mdl import Mdl
from random import random

#def Fast_1Layer_FeedForward_Mdl(): return Mdl(...ce qu'il y a dans le super)

class Fast_1Layer_FeedForward_Mdl(Mdl):
	def __init__(self, inst, required):
		inst = inst()

		inst.build_from_required(
			required, 
			inputs:=inst.need_inputs(required),
			istart:=0,
			ystart:=inputs,
			wstart:=0,
			lstart:=0,
			l2start:=0
		)
		
		#	Computing params
		_vars = inst.buildstackmodel_vars()
		weights = inst.buildstackmodel_weights()
		locds = inst.buildstackmodel_locds()
		locd2s = inst.buildstackmodel_locd2s()

		#	Labels
		_id = 0
		vsep = inst.labelstackmodel_vars(_id,ystart)
		wsep = inst.labelstackmodel_weights(_id,0)
		lsep = inst.labelstackmodel_locds(_id,0)
		l2sep = inst.labelstackmodel_locd2s(_id,0)

		#[] -> [random() for i in range(weights)]
		super().__init__([inst], inputs, _vars-inst.relativ_ystart(), _vars, weights, None, locds, locd2s, vsep, wsep, lsep, l2sep)

def Fast_Join2Mdls_FeedForward(mdl0, mdl1):
	assert mdl0.outputs == mdl1.inputs

	mdl = mdl0.copy()

	mdl.outputs = mdl1.outputs
	mdl.weights += mdl1.weights
	if mdl.w != None and mdl1.w != None:
		mdl.w += mdl1.w
	else:
		mdl.w = None

	mdl.locds += mdl1.locds
	mdl.locd2s += mdl1.locd2s
	mdl.total = mdl0.total + mdl1.total - mdl1.inputs

	new_vstart = mdl0.total - mdl0.outputs
	new_wstart = mdl0.weights
	new_lstart = mdl0.locds
	new_l2start = mdl0.locd2s

	for _plus_id,inst in enumerate(mdl1.insts):
		new_inst = inst.copy()
		i,y,w,l,l2 = new_inst.return_iywll2_start()

		i += new_vstart
		y += new_vstart
		if w != None: w += new_wstart
		if l != None: l += new_lstart
		if l2 != None: l2 += new_l2start

		new_inst.params = new_inst.setupparamsstackmodel(i,y,w,l, l2, [param for i,param in enumerate(new_inst.params) if new_inst.requiredposition[i]])

		mdl.vsep += new_inst.labelstackmodel_vars(_plus_id, y)
		mdl.wsep += new_inst.labelstackmodel_weights(_plus_id, w)
		mdl.lsep += new_inst.labelstackmodel_locds(_plus_id, l)
		mdl.l2sep += new_inst.labelstackmodel_locds(_plus_id, l2)

		mdl.insts += [new_inst]

	return mdl

def Fast_JoinNMdls_FeedForward(mdls):
	for i in range(len(mdls) - 1):
		mdls[0] = Fast_Join2Mdls_FeedForward(mdls[0], mdls[1 + i])

	return mdls[0]

def Fast_NLayers_FeedForward_Mdl(couples_inst_required):
	'''
couples_inst_required = [
	(DOT1D, [28*28,10,1,0]),
	(SOFTMAX, [10])
]
	'''
	#On join plein de Fast_1Layer_FeedForward_Mdl ensemble. On se casse pas la Tete
	#pas opti mais pas grave

	mdls = [Fast_1Layer_FeedForward_Mdl(inst, req) for inst, req in couples_inst_required]

	return Fast_JoinNMdls_FeedForward(mdls)

'''class Fast_Feed_Forward_Mdl(Mdl):
	def __init__(self, insts_required):
		"""
insts_required = [
	(Inst0, [req0, req1]),
	(Inst1, [req0, req1, req2, req3])
	...
]		"""
		insts = []
		inputs = insts_required[0][0]().need_inputs(insts_required[0][1])
		vsep, wsep, lsep = [], [], []

		istart = 0
		vars_stack = inputs
		weights_stack = 0
		locds_stack = 0

		for _id,(inst, required) in enumerate(insts_required):
			insts += [inst()]

			#inputs += [inst.need_inputs(required)]

			#vars_stack += inputs[-1]

			inst.build_from_required(
				required=required,
				inputs=inst.need_inputs(required),
				istart=istart,
				ystart=stack_start,
				wstart=weights_stack,
				lstart=locds_stack
			)

			vsep += inst.labelstackmodel_vars(   _id, vars_stack)
			wsep += inst.labelstackmodel_weights(_id, weights_stack)
			lsep += inst.labelstackmodel_locds(  _id, locds_stack)

			vars_stack += inst.buildstackmodel_vars()
			weights_stack += inst.buildstackmodel_weights()
			locds_stack += inst.buildstackmodel_locds()

			istart = vsep - inst.labelstackmodel_vars(   _id, vars_stack)

		super().__init__(insts=inst,
			inputs=inputs,
			outputs=insts[-1].buildstackmodel_vars(),
			_vars=vars_stack,
			w=[random() for i in range(weights_stack)],
			locds=locds_stack,
			vsep=vsep, wsep=wsep, lsep=lsep)

class Fast_Direct_InputOutput_Mdl(Mdl):
	def __init__(self, 
		inputs, outputs, 
		input_insts_required, corpus_insts_required, output_insts_required,
		links):

		#En fait ici on precise l'ordre des instructions (elles sont déja ordonné)
		#vars_stack = [inputs]
		istart = [0]
		ystart = []
		wstart = []
		lstart = []

		for inst in insts_required:
			#vars_stack += [vars_stack[-1] + inst.buildstackmodel_vars()]
			istart += [vars_stack[-1] + inst.buildstackmodel_vars()]
			ystart += [0]
			wstart += [0]
			lstart += [0]

		#vars_stack += [vars_stack[-1] + outputs]

		depos = [0 for i in insts_required]	#l'espace dans l'input d'une inst déjà occupé en tant que outputs d'autres fonctions

		for _form, to in links:#_from c'est l'instruction qui calcule un truc et l'ecrit dans une partie de l'input de l'instruction #to
			depos[to] += [(_from, to)]	#En fait par exemple l'instruction sum #3, va avoire un liste [(0, 3), (1, 3), (2, 3)]
										#ça veut dire que les instruction 0,1,2 déposent leurs resultats dans l'input et dans cet ordre
										#Donc le ystart du #0 sera sum.istart + 0, #1.ystart == sum.istart + #0.outputs
										#		#2.ystart == sum.istart + #0.outputs + #1.outputs
										#Et comme ça on fait une pille des resulats. Mais en réalité c'est juste l'input space vers
										#lequel les #1,#2, #3 vont écrire leur resultat l'un en dessous de l'autre.

			ystart[_from] = istart[to] + depos[to]
			depos[to] += inst[_from].buildstackmodel_vars()	#le prochain qui va se connecter a #to devra ecrire de ça plus loin

			#penser qu sum, mul qui prennent des items. Donc revoir les vars

		insts = input_insts_required + corpus_insts_required + output_insts_required
		inp_insts, corpus_insts, out_insts = len(input_insts_required), len(corpus_insts_required), len(output_insts_required)
		istart, ystart, wstart, lstart = [[0 for _ in insts] for range(4)]

		for inst,req in insts:
			#pre-build all params, juste to put them into positions (to use buildstackmodel_)
			inst.__init__()
			inst.params = inst.setupparamsstackmodel(0,0,0,0, req)

		_vars, weights, locds = [[eval(f'inst.buildstackmodel_{t}()') for inst in insts] for t in ('vars', 'weights', 'locds')]

		_tmp = 0
		for i,(inst,req) in enumerate(input_insts_required):
			_tmp += _vars[i]
			istart[i] = _tmp

		for i, (inst, req)


	LE PROBLEME EST DANS LE FAIT QUE SI 2 INSTRUCTIONS DOIVENT PASSER LEUR RESULTATS DE MANIERE UNIE VERS 2 AUTRES INSTRUCTIONS
	YSTART PEUT PAS ETRE A 2 ENDROIT EN MEME TEMPS, DONC IL FAUT ESSAYER DE GERER LE TRUC

	Imaginons 3 instruction et leurs resultat doit etre etre passé en stack 1,2,3 vers un autre Instruction A, 
	et vers une instruction B il faut les passer en mode 2,3,1    ça prend du temps et peut etre meme ajouter du code C/Cuda
	donc pour une autre fois

	La lib marche tres bien si chaque instruction peut envouer son ystart sur une seule instrcution, mais il faut verifier
	ET il faut voire pour le inputs, en gros par exemple les 3 premiere instruction peuvent prendre inputs sur istart=0
	ou sur istart=0, istart=0+c , istart=0+k    mais faut implementer ça.

class Fast_Graph_InputOutput_Mdl(Mdl):
	def __init__(self, insts_required, links):

		#on peut faire Fast_Graph_InputOutput_Mdl(Fast_Direct_InputOutput_Mdl):
		#puis un super() et donc ici on ne fait qu'ordonner et ecrire les liens de maniere structuré

		connections = []
		for this_inst, take_from_this in links:
			pass

		#All Required params have to be prebuild

		#	Les instructions qui prennent de personne prennent de l'input

		#(None for inst, required in insts_required)
		#(None for this_inst, take_from_this in links)

		#	Trouver l'Ordre des instructions

		#	Staquer les instructions et APRES construire les params (istart, ystart ...) a partire des required
		for inst in order:
			vars_stack += inst.buildstackmodel_vars()
			weights_stack += inst.buildstackmodel_weights()
			locds_stack += inst.buildstackmodel_locds()  #on peut meme faire la somme des stack directe, vu que tous les Required sont calculés

			#en fait 

		#il faut faire une 
'''
