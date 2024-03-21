from os import listdir

sable_i = max(map(lambda x:int(x[4:]), listdir('sable')))

exec(f"from sable.bac_{sable_i}.main import main")

objs = ('insts', 'scores', 'optis', 'gtics')
tous_les_objets = {obj:[] for obj in objs}

for obj in objs:
	#	Depuis le package
	exec(f"from package.package import {obj.upper()}")
	tous_les_objets[obj] += eval(obj.upper())

	#	Les fichier du sable
	i = len(eval(obj.upper()))-1
	for fichier in listdir(f'sable/bac_{sable_i}/{obj}/'):
		fichier = fichier.replace('.py', '')
		if fichier in ('__pycache__',): continue

		#	Ajout d'une classe d'un fichier
		exec(f"from sable.bac_{sable_i}.{obj}.{fichier} import {fichier.upper()}")
		tous_les_objets[obj] += [eval(fichier.upper())]
		eval(fichier.upper()).ID = i; eval(fichier.upper())._id = i
		i += 1

main(tous_les_objets)