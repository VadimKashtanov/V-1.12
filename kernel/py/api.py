from kernel.py.mdl import Mdl
from kernel.py.use import Use
from kernel.py.train import Train
from kernel.py.data import Data

from time import time

################# Des fonctions Simples Pour les Models ################################

def load_mdl(INSTS, file:str):
	#global INSTS

	mdl = Mdl(insts=[], inputs=0, outputs=0, _vars=0, weights=0, w=[], locds=0, locd2s=0, vsep=[], wsep=[], lsep=[], l2sep=[])
	with open(file, "rb") as co:
		mdl.load(co.read(), INSTS)

	return mdl

def write_mdl(mdl:Mdl, file:str):
	with open(file, "wb") as co:
		co.write(mdl.bins())

#################### Fonctions Simples Pour Les données ################################

def load_data(file:str):
	data = Data(1, 1, [0], [0])
	with open(file, "rb") as co:
		data.load(co.read())
	return data

##################### Fonction / Macro / Etc ############################

def time_taken(call, *args, **kargs):
	start = time()

	_return = call(*args, **kargs)

	print(f"La fonction {call} a pris {round(time()-start, 5)} secondes")

	return _return