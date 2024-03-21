#! /usr/bin/python3

'''
{
	'nom' : 'dot1d',

	'modes' : {	#ceci seront mis sous commentaires et desactivee si FORWARD_BACKWARD == False ou FORWARD_BACKWARD_2 == False
		'th11' : {
			'_use' : {'x' : 'Ax', 'y' : 'Bx'},
			'_forward' : {'x' : 'Ax', 'y' : 'Bx', 'sets'},

			...

			'_backward_of_forward2' : {'x' : 'Ax', 'y' : 'Bx', 'sets'}
		}
	}
}

ex _forward

for x in range(Ax):
	for y in range(Bx):
		pass

uint x = threadIdx.x + blockIdx.x * blockDim.x;
uint y = threadIdx.y + blockIdx.y * blockDim.y;
uint set = blockIdx.z;

if (x < Ax && y < Bx) {
		
};

'''

if __name__ == "__main__":
	from os import argv

	with open(argv[1], 'r') as co:
		dico = exec(co.read())

	nom = dico['nom']

	classe = __import__(f"package.insts.{nom}.py.{nom}")

	params = classe.params_names
	fb = classe.FORWARD_BACKWARD
	fb2 = classe.FORWARD_BACKWARD_2