from kernel.py.mdl import Mdl
from kernel.py.data import Data
from kernel.py.use import Use
from kernel.py.train import Train

from random import random, seed
import struct as st

class Test_OBJ:
	mdl = None
	lines = None
	
	config_score = None
	config_opti = None
	config_gtic = None

	calcule_d = None
	calcule_dd = None

	def __init__(self, SCORES, OPTIS, GTICS):
		self._bins = b''

		bins = b''

		####	Le Model et le contenair de Données		####
		mdl = self.mdl
		mdl.check()
		data = Data(
			1, self.lines, 
			[random() for _ in range(self.lines * mdl.inputs)], 
			[random() for _ in range(self.lines * mdl.outputs)])
		data.check()
		self.data = data

		#
		bins +=	mdl.bins() + st.pack('I',123)
		bins += data.bins() + st.pack('I',123)

		#### Le trainneur ####

		config_score, config_opti, config_gtic = self.config_score, self.config_opti, self.config_gtic
		if OPTIS[config_opti._id].REQUIRE_DDF:
			assert mdl.allow_ddf() and SCORES[config_score._id].ALLOW_DDF
		
		calcule_d = self.calcule_d
		calcule_dd = self.calcule_dd

		bins +=	st.pack('I', calcule_d) + st.pack('I',123)
		bins += st.pack('I', calcule_dd) + st.pack('I',123)
		
		dws = list(range(mdl.weights)) if calcule_dd else None

		bins += config_score.bins()
		bins += config_opti.bins()
		bins += config_gtic.bins()

		self.train = Train(
			mdl, data,
			config_score, config_opti, config_gtic,
			SCORES, OPTIS, GTICS,
			calcule_d,
			calcule_dd,
			dws
		)
		self.train.randomize(0)
		self.train.check()

		#bins += st.pack('I', self.calcule_dd) + st.pack('I',123)

		bins += self.train.bins('_weight') + st.pack('I', 123)

		self._bins += bins

	def check(self):
		assert self.mdl != None
		assert self.lines != None
		
		assert self.config_score != None
		assert self.config_opti != None
		assert self.config_gtic != None

		assert self.calcule_d != None
		assert self.calcule_dd != None

class Test_INST(Test_OBJ):
	def bins(self):
		bins = self._bins
		
		##	==== Use ====
		use = Use(self.mdl, self.data)
		use.set_inputs(0)
		use.forward()

		#
		bins += use.bins() + st.pack('I',123)

		# ====	Train ====
		if self.train.calcule_d:
			self.train.calculer_dSdw(0)
			bins += b''.join(self.train.bins(arr)+st.pack('I', 123) for arr in ('_weight', '_var', '_grad', '_locd', '_meand'))
		
		if self.train.calcule_dd:
			self.train.calculer_dSdwdw(0)
			bins += b''.join(self.train.bins(arr)+st.pack('I', 123) for arr in self.train.arrs_tenss.keys())

		self._bins = bins
		return bins

class Test_SCORE(Test_OBJ):
	def bins(self):
		bins = self._bins
		train = self.train

		train.set_inputs(0)			#batch=0
		train.null_for_dS()
		train.forward(start_seed:=0)

		#dloss
		train.dloss()
		bins += train.bins('_grad') + st.pack('I',123)

		#loss
		train.loss()	
		bins += train.bins('_grad') + st.pack('I',123)

		#score
		train.score()	
		bins += train.bin_score() + st.pack('I',123)	#puis on verifie que le score systeme est le bon

		####################################################

		#ddloss
		train.ddS_first_null()
		train.set_inputs()

		train.forward2(start_seed:=0)
		train.dloss()
		train.backward2(start_seed)

		#	C'est comme si on derivait plusieurs loss functions ou la loss function c'est S = (meand[w] - 0), donc d(dS/meand[w])/dmeand[w] = 1
		dw = 0
		train.ddS_second_dd_restart(dw)

		train.backward_of_backward2(dw, start_seed)
		train.ddloss()

		bins += train.bins('_dd_grad') + st.pack('I',123)

		self._bins = bins
		return bins

class Test_OPTI(Test_OBJ):
	MIN_ECHOPES = None

	def bins(self):
		bins = self._bins
		train = self.train
		opti = train.opti

		bins += st.pack('I', self.MIN_ECHOPES) + st.pack('I', 123)

		for echope in range(self.MIN_ECHOPES):
			if opti.REQUIRE_DDF:
				train.calculer_dSdwdw(0)
			else:
				train.calculer_dSdw(0)
			train.opti()

		bins += train.bins('_weight') + st.pack('I', 123)

		self._bins = bins
		return bins

class Test_GTIC(Test_OBJ):
	MIN_ECHOPES = None

	def bins(self):
		bins = self._bins
		train = self.train
		gtic = train.gtic

		bins += st.pack('I', self.MIN_ECHOPES) + st.pack('I', 123)
		
		for echope in range(self.MIN_ECHOPES):
			train.gtic()

		bins += train.bins('_weight') + st.pack('I', 123)

		self._bins = bins
		return bins

def test_package(SCORES, OPTIS, GTICS, TEST_INSTS, TEST_SCORES, TEST_OPTIS, TEST_GTICS):
	bins = b''

	for test_things in (TEST_INSTS, TEST_SCORES, TEST_OPTIS, TEST_GTICS):

		bins += st.pack('I', len(test_things))
		for TEST in test_things:
			test = TEST(SCORES, OPTIS, GTICS)
			test.check()
			bins += test.bins() + st.pack('I', 123)

	return bins