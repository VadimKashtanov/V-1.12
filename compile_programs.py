from package.programs.order import PROGRAMMES_A_COMPILER
import os

def find(chemin, terminaison='.cu'):
	list_of_files = {}
	for (dirpath, dirnames, filenames) in os.walk(chemin):
		for filename in filenames:
			if filename.endswith(terminaison):
				list_of_files[filename] = os.sep.join([dirpath, filename])
	return list_of_files

if __name__ == "__main__":
	from sys import argv
	ARGS = ' '.join(argv[1:])
	for programme in PROGRAMMES_A_COMPILER:
		print(f"[\033[35;1;41m***\033[0m] ============= PROGRAM : {programme.upper()} ============")
		fichiers_en_cu = ' '.join(find(f"package/programs/{programme}", terminaison='.cu').values())
		commande = f'nvcc {ARGS} *.o {fichiers_en_cu} -o {programme}'
		print(commande)
		os.system(commande)