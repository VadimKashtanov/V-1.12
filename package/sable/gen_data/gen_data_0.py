from kernel.py.data import Data

def gen_data_0(self, mode, config):
	if mode == 'simple':
		self.data = Data(config['batchs'], config['lines'], config['input'], config['output'])

	elif mode == 'serie':
		#[1,2,3,4,5,6,7,8]
		#[1,2] -> [3,4], [3,4] -> [5,6], [5,6] -> [7,8]
		#[ [[]->[]->[]], [[]->[]->[]] ]
		serie = config['serie']
		assert type(serie) in (list,tuple)
		assert len(serie) >= 2
		assert all(type(elm)==float for elm in serie)

		paquets = config['batchs']
		X = config['X']

		assert len(serie) % X == 0

		couples = [[serie[i*X+j] for j in range(X)] for i in range(int(len(serie)/X))]
		assert (len(couples)-1) % paquets == 0
		lignes = couples_par_paquet = int(len(couples) / paquets)-1
		if couples_par_paquet == 1:
			ERR("Il n'y a qu'un coupe par paquet (batch)")

		_input = []
		output = []
		for c in range(len(couples)-1):
			_input += couples[c]
			output += couples[c+1]

		self.data = Data(
			paquets,
			lignes,
			_input,
			output,
		)
	else:
		raise Exception(f"Le mode {mode} n'existe pas.")