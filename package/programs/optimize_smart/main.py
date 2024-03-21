#
# python3 -m package.programs.optimize_mdl.compile "path to file with the class" 
#

'''

for echopes:
	for no_test_pass:
		start_seed = rand() % 10000
		
		for same_start_seed_runs:
			batchs = rand() % batchs
			set_input()
			
			for repeats:
				nullgrad()
				forward()
				dloss()
				backward()
				update()

	test()
'''


'''
{
	"mdl file" : "",
	"data file" : "",
	"test data file" : "",

	"echopes" : 0
}
'''

import struct as st

string = lambda _str: st.pack('I', len(_str)) + _str.encode()
uint = lambda uint: st.pack('I', uint)
dictionnary = lambda _dict: uint(len(_dict)) + b''.join(map(string, _dict.keys())) + b''.join(map(string, _dict.values()))
sfloat = lambda sfloat: st.pack('f', sfloat)

default = {	#ex : no mdl_file because it's required and no default exists
	'echopes' : 1,
	'no_test_passs' : 1,
	'same_start_seed_runs' : 1,
	'repeats' : 1,

	'sets' : 0,
	'is_train_random' : 1,

	'limit_score' : 0.0,

	'opti' : 0,
	'opti_args' : {},

	'score' : 0,
	'score_args' : {},

	'echo_weights' : 0,
	'echo_vars' : 0,
	'echo_locds' : 0,
	'echo_grads' : 0,
	'echo_meands' : 0
}

if __name__ == "__main__":
	from sys import argv

	with open(argv[1], "r") as co:
		exec(f"config = {{ **default, **{co.read()}}}")#merge defautl and argv[1] (and replace with the argv1 values)

	elements = {
		'mdl_file' : string,
		'data_file' : string,
		#'test_data_file' : string,
		'out_file' : string,

		'echopes' : uint,
		'no_test_passs' : uint,
		'same_start_seed_runs' : uint,
		'repeats' : uint,

		'sets' : uint,
		'is_train_random' : uint,

		'limit_score' : sfloat,

		'opti' : uint,
		'opti_args' : dictionnary,

		'score' : uint,
		'score_args' : dictionnary,

		'echo_weights' : uint,
		'echo_vars' : uint,
		'echo_locds' : uint,
		'echo_grads' : uint,
		'echo_meands' : uint
	}

	ordre = [
		'mdl_file', 'data_file', 'out_file',
		'echopes', 'no_test_passs', 'same_start_seed_runs', 'repeats',
		'sets', 'is_train_random',
		'limit_score',
		'opti', 'opti_args',
		'score', 'score_args',
		'echo_weights', 'echo_vars', 'echo_locds', 'echo_grads', 'echo_meands'
	]

	with open("package/programs/optimize_smart/tmpt", "wb") as co:
		#co.write(b''.join(tobins(config[element]) for element,tobins in elements.items()))
		co.write(b''.join(elements[element](config[element]) for element in ordre))