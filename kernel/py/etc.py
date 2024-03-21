#https://en.wikipedia.org/wiki/Pseudorandom_number_generator#Implementation

from math import exp, atan, tanh, cos, sin, pi, tan
from random import random, randint, seed
import struct as st

gauss = lambda x: exp(-(x)**2)

A_SMALL_PRIME_0 = 2017
A_SMALL_PRIME_0_1 = A_SMALL_PRIME_0 - 1
A_SMALL_PRIME_1 = 4877

#seed**3
#113*113=12769
pseudo_randomf = lambda seed: (12769*((seed*seed%12345)+1) % 4877)/4877
pseudo_randomf_minus1_1 = lambda seed: 2*pseudo_randomf(seed) - 1

def ERR(msg):
	raise Exception("\033[101m" + msg + "\033[0m")

def null(arr):
	for i in range(len(arr)):
		arr[i] = 0

def nulls(*arrs):
	for arr in arrs:
		null(arr)

################################# Bins #######################################
ftoi = float_as_uint = lambda flt: st.unpack('I', st.pack('f', flt))[0]
itof = utof = uint_as_float = lambda uint: st.unpack('f', st.pack('I', uint))[0]

def read(size, bins):
	ret = st.unpack(size, bytes(bins[:st.calcsize(size)]))
	del bins[:st.calcsize(size)]
	return ret

st_123 = st.pack('I', 123)

def lire_123(bins):
	if not read('I', bins) == 123: ERR("")

def read_str(size, bins):
	_len = read('I', bins)
	return  b''.join(read('c'*_len, bins)).decode()

def write_str(str):
	bins = st.pack('I', len(str))
	bins += str.encode()
	return bins

########################### Simple Terminal Plotting #########################

def terminale_plateau_pixels(platxy):
	max_len = max(len(td) for td, _, _ in platxy)
	for (text_debut, ligney, text_apres) in platxy:
		print(text_debut + " "*(max_len-len(text_debut)), end='\033[100;94;4m ')
		for pix,digit0,digit1 in ligney:
			_0_ou_1 = pix
			coul = [100, 107][_0_ou_1]
			print(f"\033[{coul};94;4m{digit0}{digit1}\033[100;4m ", end='')
		print(f"\033[0m {text_apres}")

def term_plot(inp_arr, H=20, print_pos=True):
	h = H - 1
	print(f"Plotting {inp_arr}")
	if print_pos:
		if not len(inp_arr) <= 100: ERR("")
	_max = max(inp_arr)
	_min = min(inp_arr)

	if _max == _min:
		term_plot_memes_listes(inp_arr, H, print_pos)
		return
	
	arr = []
	for i in range(len(inp_arr)):
		arr += [(inp_arr[i] - _min)/(_max - _min)]

	plateau = [["", [(0, ' ', ' ') for i in range(len(arr))], ""] for y in range(H)]

	for i in range(len(arr)):
		#print(arr[i]*h)
		hauteur = H - int(arr[i]*h) - 1
		_str = list(str(i))
		if len(_str) == 1: _str = ['0', _str[0]]
		if print_pos: plateau[hauteur][1][i] = (1, _str[0], _str[1])
		else: plateau[hauteur][1][i] = (1, ' ', ' ')

		for j in range(hauteur+1, H):
			plateau[j][1][i] = (1, ' ', ' ')
	for y in range(H):
		plateau[y][0] = str(round(_max - (_max-_min)*y/h, 5))
		
	terminale_plateau_pixels(plateau)

def term_plot_memes_listes(inp_arr, H=20, print_pos=True):
	H = 1
	h = H - 1
	_max = max(inp_arr)
	_min = min(inp_arr)

	arr = []
	for i in range(len(inp_arr)):
		arr += [(inp_arr[i] - _min)]

	plateau = [["", [(0, ' ', ' ') for i in range(len(arr))], ""] for y in range(H)]

	for i in range(len(arr)):
		#print(arr[i]*h)
		hauteur = H - int(arr[i]*h) - 1
		_str = list(str(i))
		if len(_str) == 1: _str = ['0', _str[0]]
		if print_pos: plateau[hauteur][1][i] = (1, _str[0], _str[1])
		else: plateau[hauteur][1][i] = (1, ' ', ' ')

		for j in range(hauteur+1, H):
			plateau[j][1][i] = (1, ' ', ' ')
	for y in range(H):
		plateau[y][0] = str(round(_max, 5))
		
	terminale_plateau_pixels(plateau)