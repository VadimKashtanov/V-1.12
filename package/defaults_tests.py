from kernel.py.config import Config
from kernel.py.etc import *

default_test_score = Config(0, [])
default_test_opti = Config(0, [uint_alpha:=float_as_uint(0.1)])
default_test_gtic = Config(0, [sets:=3])