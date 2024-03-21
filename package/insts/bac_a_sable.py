from kernel.py.inst import Inst
from kernel.py.etc import *

#En plus des simple_d_xhy.py simple_f_xy.py
#Il y a ca qui des le depart ne calcule pas soit ddf soit df&ddf

class Non_Impl_ddF(Inst):
	FORWARD_BACKWARD_2 = True

	def check(self): pass
	def check_model(self, insts_ids:[int], params:[[int]], this_inst_pos:int): pass

	def forward2(self, start_seed:int,
		sets:int, vsize:int, wsize:int, lsize:int, l2size:int, _set:int, time:int,
		weight:[float], var:[float],
		locd:[float], grad:[float], meand:[float],
		dd_w:[float], dd_var:[float],
		dd_locd:[float], dd_grad:[float], dd_meand:[float]):
		assert ERR("ddF est desactive sur cette instruction")

	def backward2(self, start_seed:int,
		sets:int, vsize:int, wsize:int, lsize:int, l2size:int, _set:int, time:int,
		weight:[float], var:[float],
		locd:[float], grad:[float], meand:[float],
		dd_w:[float], dd_var:[float],
		dd_locd:[float], dd_grad:[float], dd_meand:[float]):
		assert ERR("ddF est desactive sur cette instruction")

	def backward_of_backward2(self, start_seed:int,
		dw:int,	#Matrice hessienne c'est d(dS/dw)/d[] on fait ligne par ligne (plus opti)
		sets:int, vsize:int, wsize:int, lsize:int, l2size:int, _set:int, time:int,
		weight:[float], var:[float],
		locd:[float], grad:[float], meand:[float],
		locd2:[float],
		dd_w:[float], dd_var:[float],
		dd_locd:[float], dd_grad:[float], dd_meand:[float]):
		assert ERR("ddF est desactive sur cette instruction")

	def backward_of_forward2(self, start_seed:int,
		dw:int,	#Matrice hessienne c'est d(dS/dw)/d[] on fait ligne par ligne (plus opti)
		sets:int, vsize:int, wsize:int, lsize:int, l2size:int, _set:int, time:int,
		weight:[float], var:[float],
		locd:[float], grad:[float], meand:[float],
		locd2:[float],
		dd_w:[float], dd_var:[float],
		dd_locd:[float], dd_grad:[float], dd_meand:[float]):
		assert ERR("ddF est desactive sur cette instruction")

class Non_Impl_dF_ddF(Inst, Non_Impl_ddF):
	FORWARD_BACKWARD = True