from package.insts.fast_model import Fast_JoinNMdls_FeedForward
from package.package import INSTS
from kernel.py.api import load_mdl, write_mdl

if __name__ == "__main__":
	from sys import argv

	del argv[0] 

	mdls = []

	for file in argv[1:]:
		mdls += [load_mdl(INSTS, file)]

	write_mdl(Fast_JoinNMdls_FeedForward(mdls), argv[0])