############################################### INSTS ##############################################

from package.insts.order import INSTS_ORDER

INSTS = []
INSTS_DICT = {}

for _id, elm in enumerate(INSTS_ORDER):
	exec(f"from package.insts.{elm.lower()}.py.{elm.lower()} import {elm.upper()}")
	exec(f"INSTS += [{elm.upper()}]")
	exec(f"INSTS_DICT[f'{elm.upper()}'] = {elm.upper()}")
	exec(f"INSTS_DICT[f'{elm}'] = {elm.upper()}")
	exec(f"{elm.upper()}._id = {_id}")
	exec(f"{elm.upper()}.ID = {_id}")

############################################## SCORES ##############################################

from package.scores.order import SCORES_ORDER

SCORES = []
SCORES_DICT = {}

for _id, elm in enumerate(SCORES_ORDER):
	exec(f"from package.scores.{elm.lower()}.py.{elm.lower()} import {elm.upper()}")
	exec(f"SCORES += [{elm.upper()}]")
	exec(f"SCORES_DICT[f'{elm.upper()}'] = {elm.upper()}")
	exec(f"SCORES_DICT[f'{elm}'] = {elm.upper()}")
	exec(f"{elm.upper()}._id = {_id}")
	exec(f"{elm.upper()}.ID = {_id}")

############################################### OPTIS ##############################################

from package.optis.order import OPTIS_ORDER

OPTIS = []
OPTIS_DICT = {}

for _id, elm in enumerate(OPTIS_ORDER):
	exec(f"from package.optis.{elm.lower()}.py.{elm.lower()} import {elm.upper()}")
	exec(f"OPTIS += [{elm.upper()}]")
	exec(f"OPTIS_DICT[f'{elm.upper()}'] = {elm.upper()}")
	exec(f"OPTIS_DICT[f'{elm}'] = {elm.upper()}")
	exec(f"{elm.upper()}._id = {_id}")
	exec(f"{elm.upper()}.ID = {_id}")

############################################### GTICS ##############################################

from package.gtics.order import GTICS_ORDER

GTICS = []
GTICS_DICT = {}

for _id, elm in enumerate(GTICS_ORDER):
	exec(f"from package.gtics.{elm.lower()}.py.{elm.lower()} import {elm.upper()}")
	exec(f"GTICS += [{elm.upper()}]")
	exec(f"GTICS_DICT[f'{elm.upper()}'] = {elm.upper()}")
	exec(f"GTICS_DICT[f'{elm}'] = {elm.upper()}")
	exec(f"{elm.upper()}._id = {_id}")
	exec(f"{elm.upper()}.ID = {_id}")

