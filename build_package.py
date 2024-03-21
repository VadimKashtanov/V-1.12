#! /usr/bin/python3

def implementer_liste(_type:str, nom:str, nombre:str, liste_noms:[str], fin:str='\n'):
	j = '\t' if fin == '\n' else ''
	n = '\t' if fin != '\n' else ''
	return f"{_type} {nom}[{nombre}] = {{" + '\n' + \
		n+fin.join( f"{j}{elm_nom}" for elm_nom in liste_noms) + \
	"\n};\n"

def liste_nom_definitions_et_commentaire(paterne, liste_elements:[str]):
	return [paterne.format(elm) + f',\t//{elm}' for elm in liste_elements]

def hastag_sep(mot:str, taille=100, c=False):
	l = 100 - len(mot)
	symbole = '=' if c else '#'
	return ('//' if c else '') + symbole*int(l/2+0.5) + mot + symbole* int(l/2)

########################################################################################################
####################### Tous les Objets configurables de la version 1.9 ################################
########################################################################################################

OBJS = 'inst', 'score', 'opti', 'gtic'

for obj in OBJS:
	exec(f'from package.{obj}s.order import {obj.upper()}S_ORDER')

	for elm in eval(f'{obj.upper()}S_ORDER'):
		exec(f'from package.{obj}s.{elm}.py.{elm} import {elm.upper()}')

	#	On verifie que les objets ne portent pas le meme nom (localement relativement aux obj ci)
	assert all(eval(obj.upper()+'S_ORDER').count(elm) == 1 for elm in eval(obj.upper()+'S_ORDER'))

#	On verifie que les differents objets ne partagent pas des noms identiques entre eux
TOUS = [elm.upper() for obj in OBJS for elm in eval(obj.upper()+'S_ORDER')]
for elm in TOUS:
	assert all(TOUS.count(elm) == 1 for elm in TOUS)

#	Quelques Conditions de convention afin de simplifier le debbug et de maniere generale tout
assert eval(INSTS_ORDER[0].upper()).ALLOW_DF == 1
assert eval(INSTS_ORDER[0].upper()).ALLOW_DDF == 1
assert eval(SCORES_ORDER[0].upper()).ALLOW_DDF == 1

########################################################################################################
################################## PACKAGE.PY && TEST_PACKAGE.PY #######################################
########################################################################################################

package_py = ''

for obj in OBJS:
	package_py += hastag_sep(f' {obj.upper()}S ') + 2*'\n'

	package_py += f'from package.{obj}s.order import {obj.upper()}S_ORDER' + 2*'\n'
	obj_order = f"{obj.upper()}S_ORDER"

	package_py += f'{obj.upper()}S = []' + '\n'
	package_py += f'{obj.upper()}S_DICT = {{}}' + '\n'
	package_py += '\n'
	
	package_py += f"""for _id, elm in enumerate({obj_order}):
	exec(f"from package.{obj}s.{{elm.lower()}}.py.{{elm.lower()}} import {{elm.upper()}}")
	exec(f"{obj.upper()}S += [{{elm.upper()}}]")
	exec(f"{obj.upper()}S_DICT[f'{{elm.upper()}}'] = {{elm.upper()}}")
	exec(f"{obj.upper()}S_DICT[f'{{elm}}'] = {{elm.upper()}}")
	exec(f"{{elm.upper()}}._id = {{_id}}")
	exec(f"{{elm.upper()}}.ID = {{_id}}")

"""

with open("package/package.py", "w") as co:
	co.write(package_py)

### Test Package

test_package_py = f"from package.package import {', '.join(obj.upper()+'S_ORDER' for obj in OBJS)}" + '\n'

for obj in OBJS:
	test_package_py += f"""
{hastag_sep(f' TESTE {obj.upper()}S ')}

TEST_{obj.upper()}S = []
PAPIER_{obj.upper()}S = []
for elm in {obj.upper()}S_ORDER:
	exec(f"from package.{obj}s.{{elm}}.py.test_{obj}_{{elm}} import TEST_{obj.upper()}_{{elm.upper()}}, PAPIER_{obj.upper()}_{{elm.upper()}}")
	exec(f"TEST_{obj.upper()}S += [TEST_{obj.upper()}_{{elm.upper()}}]")
	exec(f"PAPIER_{obj.upper()}S += [PAPIER_{obj.upper()}_{{elm.upper()}}]")
"""

test_package_py += "\n" + f"TESTS = [{', '.join(f'TEST_{obj.upper()}S' for obj in OBJS)}]" + "\n"
test_package_py += f"PAPIERS = [{', '.join(f'PAPIER_{obj.upper()}S' for obj in OBJS)}]"

with open("package/test_package.py", "w") as co:
	co.write(test_package_py)

########################################################################################################
############################################ META.CUH ##################################################
########################################################################################################

meta_cuh = f"#pragma once\n\n" + '\n'.join(f'#define {obj.upper()}S {len(eval(f"{obj.upper()}S_ORDER"))}' for obj in OBJS)

with open("package/meta.cuh", "w") as co:
	co.write(meta_cuh)

########################################################################################################
############################################ PACKAGE.CUH ###############################################
########################################################################################################

package_cuh = f"""#pragma once

//	Pour package/ et head/, dans des #define la quantité d'objs
#include "package/meta.cuh"

//	Ce .cuh inclue tous les .cuh du head/
#include "kernel/head/testpackage.cuh"

//	Chaque obj a un .cuh ou toutes les (*)[] sont stoqué
""" + '\n'.join(f'#include "package/{obj}s/{obj}s.cuh"' for obj in OBJS) + "\n\n//	Arrays are declared in headers and writed in package/src/*.cu"

with open("package/package.cuh", "w") as co:
	co.write(package_cuh)

########################################################################################################
############################################ OBJ.CUH ###################################################
########################################################################################################

from package.package import INSTS, SCORES, OPTIS, GTICS

valeurs = 'ID', 'name', 'params_names', 'ALLOW_DDF', 'REQUIRE_DDF'

for objs in INSTS, SCORES, OPTIS, GTICS:
	for elm in objs:
		attributs = dir(elm)
		for val in valeurs:
			if val in attributs: assert eval(f"elm.{val}") != None

cuh = {
	f'{obj}' : '#pragma once\n\n' + '\n'.join(f'#include "package/{obj}s/{elm}/head/{elm}.cuh"' for elm in eval(f'{obj.upper()}S_ORDER'))
		for obj in OBJS
}

cu = {
	f'{obj}' : f'#include "package/{obj}s/{obj}s.cuh"\n\n'
		for obj in OBJS
}

#	Tous les objets sont structuré par un Config_t, donc ont tous des parametres et un ID
#

#je pourrais faire dans arrays[obj] *liste_nom_paramettres_par_objet
for obj in OBJS:
	cu[obj] += hastag_sep(' Des Constantes implémenté ici car plus simple ', c=True) + '\n\n'

	#static const char obj_nom_params_noms[N] = {
	#	"p0", "p1", "p2" ...
	#}
	liste_nom_paramettres_par_objet = [
		['static const char*', f'{elm.name.lower()}_params_names', f'{len(elm.params_names)}', ('"'+e+'"' for e in elm.params_names), ', ']
			for elm in eval(f'{obj.upper()}S')
	]

	for _type, nom, nombre, liste_noms, _fin in liste_nom_paramettres_par_objet:
		cu[obj] += implementer_liste(_type, nom, nombre, liste_noms, _fin) + '\n'

	if obj != 'inst':
		liste_valeurs_default_paramettres_par_objet = [
			['static const uint', f'{elm.name.lower()}_params_defaults', f'{len(elm.params_names)}', elm.params_defaults, ', ']
				for elm in eval(f'{obj.upper()}S')
		]

		for _type, nom, nombre, liste_noms, _fin in liste_valeurs_default_paramettres_par_objet:
			cu[obj] += implementer_liste(_type, nom, nombre, liste_noms, _fin) + '\n'

	cu[obj] += hastag_sep(' Toutes les arrays de fonctions ou de constantes ', c=True) + '\n\n'

#	Se referer a kernel/head/toutes_les_arrays.txt pour voire toutes les arrays qu'il faut implementer
#

arrays = {
	'inst' : [
		###
		['uint', 'INST_params', 'INSTS', [f'{len(inst.params_names)}, //{inst.name}' for inst in INSTS]],
		['const char*', 'INST_name', 'INSTS', ('"'+e+'"' for e in INSTS_ORDER)],
		['const char**', 'INST_param_name', 'INSTS',  liste_nom_definitions_et_commentaire('{}_params_names', INSTS_ORDER)],
		['uint', 'INST_capable_df', 'INSTS', [f'{int(eval(f"{inst.upper()}.ALLOW_DF"))}, //{inst}' for inst in INSTS_ORDER]],
		['uint', 'INST_capable_ddf', 'INSTS', [f'{int(eval(f"{inst.upper()}.ALLOW_DDF"))}, //{inst}' for inst in INSTS_ORDER]],

		###
		['inst_check_f', 'INST_CHECK', 'INSTS', liste_nom_definitions_et_commentaire('{}_check', INSTS_ORDER)],
		['cpu_f', 'INST_CPU', 'INSTS', liste_nom_definitions_et_commentaire('{}_cpu', INSTS_ORDER)],
		['use_f', 'INST_USE', 'INSTS', liste_nom_definitions_et_commentaire('{}_use', INSTS_ORDER)],

		['train_f', 'INST_FORWARD', 'INSTS', liste_nom_definitions_et_commentaire('{}_forward', INSTS_ORDER)],
		['train_f', 'INST_BACKWARD', 'INSTS', liste_nom_definitions_et_commentaire('{}_backward', INSTS_ORDER)],

		['train_f', 'INST_FORWARD2', 'INSTS', liste_nom_definitions_et_commentaire('{}_forward2', INSTS_ORDER)],
		['train_f', 'INST_BACKWARD2', 'INSTS', liste_nom_definitions_et_commentaire('{}_backward2', INSTS_ORDER)],
		['train_dw_f', 'INST_BACKWARD_OF_BACKWARD2', 'INSTS', liste_nom_definitions_et_commentaire('{}_backward_of_backward2', INSTS_ORDER)],
		['train_dw_f', 'INST_BACKWARD_OF_FORWARD2', 'INSTS', liste_nom_definitions_et_commentaire('{}_backward_of_forward2', INSTS_ORDER)],
	],

	'score' : [
		##
		['const char*', 'SCORE_name', 'SCORES',  ('"'+e+'"' for e in SCORES_ORDER)],
		['const uint', 'SCORE_params', 'SCORES', [f'{len(score.params_names)}, //{score.name}' for score in SCORES]],
		['const char**', 'SCORE_params_names', 'SCORES',  liste_nom_definitions_et_commentaire('{}_params_names', SCORES_ORDER)],
		['const uint*', 'SCORE_defaults', 'SCORES',  liste_nom_definitions_et_commentaire('{}_params_defaults', SCORES_ORDER)],

		['const uint', 'SCORE_allow_ddf', 'SCORES', [f'{int(score.ALLOW_DDF)}, //{score.name}' for score in SCORES]],

		##
		['dict_config_f', 'SCORE_STR_CONFIG', 'SCORES', liste_nom_definitions_et_commentaire('{}_str_config', SCORES_ORDER)],
		['func_train_f', 'SCORE_MK', 'SCORES', liste_nom_definitions_et_commentaire('{}_mk', SCORES_ORDER)],

		['func_train_f', 'SCORE_SCORE', 'SCORES', liste_nom_definitions_et_commentaire('{}_score', SCORES_ORDER)],
		['func_train_f', 'SCORE_LOSS', 'SCORES', liste_nom_definitions_et_commentaire('{}_loss', SCORES_ORDER)],
		['func_train_f', 'SCORE_DLOSS', 'SCORES', liste_nom_definitions_et_commentaire('{}_dloss', SCORES_ORDER)],
		['func_train_f', 'SCORE_DDLOSS', 'SCORES', liste_nom_definitions_et_commentaire('{}_ddloss', SCORES_ORDER)],

		['func_train_f', 'SCORE_FREE', 'SCORES', liste_nom_definitions_et_commentaire('{}_free', SCORES_ORDER)],
	],

	'opti' : [
		##
		['const char*', 'OPTI_name', 'OPTIS',  ('"'+e+'"' for e in OPTIS_ORDER)],
		['const uint', 'OPTI_params', 'OPTIS', [f'{len(opti.params_names)}, //{opti.name}' for opti in OPTIS]],
		['const char**', 'OPTI_params_names', 'OPTIS',  liste_nom_definitions_et_commentaire('{}_params_names', OPTIS_ORDER)],
		['const uint*', 'OPTI_defaults', 'OPTIS',  liste_nom_definitions_et_commentaire('{}_params_defaults', OPTIS_ORDER)],

		['const uint', 'OPTI_require_ddf', 'OPTIS', [f'{int(opti.REQUIRE_DDF)}, //{opti.name}' for opti in OPTIS]],

		##
		['dict_config_f', 'OPTI_STR_CONFIG', 'OPTIS', liste_nom_definitions_et_commentaire('{}_str_config', OPTIS_ORDER)],
		['func_train_f', 'OPTI_MK', 'OPTIS', liste_nom_definitions_et_commentaire('{}_mk', OPTIS_ORDER)],

		['func_train_f', 'OPTI_OPTI', 'OPTIS', liste_nom_definitions_et_commentaire('{}_opti', OPTIS_ORDER)],

		['func_train_f', 'OPTI_FREE', 'OPTIS', liste_nom_definitions_et_commentaire('{}_free', OPTIS_ORDER)],
	],

	'gtic' : [
		##
		['const char*', 'GTIC_name', 'GTICS',  ('"'+e+'"' for e in GTICS_ORDER)],
		['const uint', 'GTIC_params', 'GTICS', [f'{len(gtic.params_names)}, //{gtic.name}' for gtic in GTICS]],
		['const char**', 'GTIC_params_names', 'GTICS',  liste_nom_definitions_et_commentaire('{}_params_names', GTICS_ORDER)],
		['const uint*', 'GTIC_defaults', 'GTICS',  liste_nom_definitions_et_commentaire('{}_params_defaults', GTICS_ORDER)],

		##
		['dict_config_f', 'GTIC_STR_CONFIG', 'GTICS', liste_nom_definitions_et_commentaire('{}_str_config', GTICS_ORDER)],
		['func_train_f', 'GTIC_MK', 'GTICS', liste_nom_definitions_et_commentaire('{}_mk', GTICS_ORDER)],

		['func_train_f', 'GTIC_GTIC', 'GTICS', liste_nom_definitions_et_commentaire('{}_gtic', GTICS_ORDER)],

		['func_train_f', 'GTIC_FREE', 'GTICS', liste_nom_definitions_et_commentaire('{}_free', GTICS_ORDER)],
	]
}
#	On peut evidement simplifier. Le paterne est simple.
# for obj in OBJS: arrays[obj] += ['const char*', f'{obj.upper()}_name', ...] ... et on fait une liste des choses en ajoutant, quels objets l'ont

for obj in OBJS:
	for _type, nom, nombre, liste_noms in arrays[obj]:
		cu[obj] += implementer_liste(_type, nom, nombre, liste_noms) + '\n'

#	A la fin, comme ça si il y a des erreurs, on ecrit pas cuh
for obj in OBJS:
	with open(f"package/{obj}s/{obj}s.cu", "w") as co:
		co.write(cu[obj])

	with open(f"package/{obj}s/{obj}s.cuh", "w") as co:
		co.write(cuh[obj])