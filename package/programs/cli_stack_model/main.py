from package.package import INSTS, INSTS_DICT
from kernel.py.mdl import Mdl
from random import random
from os import system

#
#	On remarquera que tous les arguments en int's sont lu avec int(eval()) et pas juste int()
#	Tout simplement car ca reste du python, et avec la commande "py" on execute du python
#	Donc je peux faire "py A = 5/2" et puis "inputs A" et les inputs seront int(5/2) donc 2
#	On peut faire avec plusieurs lignes de commande	
#

from package.insts.activations import activ_names

mots_possibles = {**activ_names}
mots_possibles_keys = mots_possibles.keys()

'''
	Methode:

		User select insts list and Required Params
			|
			v
		Package Build Real Params with istart/ystart/wstart/lstart and Separators (Labels) and Compile Into Bin file
'''

'''
exemple:

>>> add dot1d Ax=32 drop_rate=20 ...
'''

def give_help():
	print("""Principe:
Il faut au moins une instruction et set les inputs.
Pour les parametres il faut mettre que les required.
C'est un model de stacking c'est facile de calculer les istart,ystart ...
Les vsep,wsep et lsep vont être mis en plus automatiquement aussi.

All commands:
	help 		Print Help
	add 		Add instruction to end of stack "add <inst_name> <param0=int> <param1=int> ..."
	del 		Del last instruction
	exit		Exit without saving
	compile		Compile into file	"compile <file>"
	print		Print the liste of actual instructions
	inputs		Set the amount of inputs
	clear		clear the screen
	breakpoint	to debbug

All package Instructions:
		""")

	for inst in INSTS:
		print(f"	{inst._id} | {inst.name}, Params : {inst.params_names}. Required for building stack : {inst.requiredforsetupparams}")

insts = []
inputs = None
echo = False

def command(line):
	global insts, inputs, echo

	word = line.strip().replace('  ', ' ').split(' ')

	if echo:
		print("\033[3m"+line+"\033[0m")

	if word == []:
		pass

	elif word[0][0] == '#':
		pass

	elif word[0].lower() == 'echo':
		if word[1].lower() == 'on':
			echo = True
		elif word[1].lower() == 'off':
			echo = False
		else:
			print(f"\033[91m Qu'es 'echo {word[1]}' ?")

	elif word[0].lower() in ('cls', 'clear'):
		system("clear")
	
	elif word[0].lower() == "breakpoint":
		breakpoint()

	elif word[0].lower() in ('help()', 'help', 'h'):
		give_help()

	elif word[0].lower() == 'del':
		if len(insts) == 0:
			print("\033[91m Il n'y a aucune instruction \033[0m")
		else:
			del insts[-1]

	elif word[0].lower() == 'print':
		for i,inst in enumerate(insts):
			print(f"{i} | {inst.name} " + " ".join(r + '=' + str(inst.params[inst.params_names.index(r)]) for r in inst.requiredforsetupparams))

	elif word[0].lower() == 'py':
		cmd = ' '.join(word[1:])
		exec(cmd, globals())

	elif word[0].lower() == 'add':
		inst_name = word[1].upper()

		try:
			inst = INSTS_DICT[inst_name]
		except KeyError:
			print(f"\033[101m No {inst_name} instruction. (breakpoint to debbug) \033[0m")
			return False

		str_params_names, str_params_values = list(zip(*list(map(lambda x:x.split('='), word[2:]))))

		params = []
		#problem = False

		#On construit tous les paramtres (istart, ystart ... aussi) un par un, et on ira chercher l'info a chaue fois
		for p,is_required in enumerate(inst.requiredposition):
			#If it's required, we try to read from command line
			#otherwise it will be build in "compile" part (because it work this way in cli_stack_model)
			if is_required == 1:
				if inst.params_names[p] in str_params_names:
					#	The required is in the command line, we read it
					argument = str_params_values[str_params_names.index(inst.params_names[p])]

					if argument in mots_possibles_keys: valeur_uint = mots_possibles[argument]
					else: valeur_uint = int( eval(argument) )

					assert valeur_uint >= 0

					params += [valeur_uint]

				else:
					#	Nous ne trouvons pas le parametre requis de l'instruction
					if inst.params_names[p] in inst.params_defaults:
						params += [inst.params_defaults[inst.params_names[p]]]
					else:
						print(f"\033[91m Ou est <{inst.params_names[p]}> (aucunes valeurs par default disponibles) ? \033[0m")
						return False
			else:
				params += [None]

		#if problem == False:
		insts += [inst(params)]

	elif word[0].lower() == 'inputs':
		inputs = int(eval(word[1]))

	elif word[0].lower() == 'compile':
		file = word[1]

		if inputs == None:
			print(f"\033[91m Les inputs n'ont pas été mis en place (voir help pour plus de renseignement ? \033[0m")
			return False

		_vars = 0
		_weights = 0
		_locds = 0
		_locd2s

		istart = 0
		ystart = inputs
		wstart = 0
		lstart = 0
		l2start = 0

		vsep = []
		wsep = []
		lsep = []
		l2sep = []

		last_vars = inputs
		last_relativ_ystart = 0

		#inst_number == line == layer number == instruction position in model
		#inst == inst class
		for inst_number, inst in enumerate(insts):
			this_vars = inst.buildstackmodel_vars()
			this_weights = inst.buildstackmodel_weights()
			this_locds = inst.buildstackmodel_locds()
			this_locd2s = inst.buildstackmodel_locd2s()

			required = [val for i,val in enumerate(inst.params) if inst.requiredposition[i]]

			try:
				#verifier que la taille de l'output de la precedante instruction est la meme que le input necessaire selon required params
				inst.check_input_output(last_vars-last_relativ_ystart, required)
			except:
				raise Exception(f"\033[101mError on line {inst_number}, with {inst.name} line \033[0m")

			inst.params = inst.setupparamsstackmodel(istart+last_relativ_ystart, ystart, wstart, lstart, required)

			#	Check here because even if this_var is incorrect, this will stop all
			inst.check()

			#ces fonctions retournes deja des listes
			vsep += inst.labelstackmodel_vars(inst_number, ystart)
			wsep += inst.labelstackmodel_weights(inst_number, wstart)
			lsep += inst.labelstackmodel_locds(inst_number, lstart)
			l2sep += inst.labelstackmodel_locds(inst_number, l2start)

			#	Positions for next instruction
			istart += last_vars #in case of LSTM istart have to start at .h not .e (witch is also _vars)
			ystart += this_vars
			wstart += this_weights
			l2start += this_locd2s

			#	Global amount of floats that need to be alloced
			_vars += this_vars
			_weights += this_weights
			_locds += this_locds
			_locd2s += this_locd2s
	
			last_vars = this_vars
			last_relativ_ystart = inst.relativ_ystart()

		outputs = last_vars - last_relativ_ystart

		with open(file, 'wb') as co:
			mdl = Mdl(
				#	List of instructions
				insts,

				inputs,
				outputs,

				_vars,

				_weights,
				[random() for w in range(_weights)],
				_locds,

				vsep,
				wsep,
				lsep,
				l2sep

			)

			mdl.check()

			

			co.write(mdl.bins())

		exit()

	elif word[0].lower() in ('exit', 'q', 'quit', 'exit()', 'quit()'):
		exit()

	else:
		print(f"Qu'es que cette commands ? : {word[0]}")

	return True

def compile_text(lines):
	for line in lines:
		if not line.replace(' ', '').replace('\t', '') == "":
			if not command(line):#.strip().replace('  ', ' ').split(' '))
				print("Abropting compilation")
				return False
	return True

if __name__ == "__main__":
	from sys import argv

	if len(argv) == 2:
		with open(argv[1], 'r') as co:
			lines = co.read().split('\n')

			assert compile_text(lines) == True

	else:
		print("""Simple Machine Learning 1.0.
Cli application to build stack/layers models with all package instructions.
First Version of Cli (2022). Simplification of Tkinter_stack_model.
	help() and quit() to get help and quit this program.
""")
		while True:
			line = input(">>> ");#.strip().replace('  ', ' ').split(' ')
			if command(line) == False:
				print("\033[101m Error \033[0m")