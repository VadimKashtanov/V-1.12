from package.package import INSTS_ORDER, SCORES_ORDER, OPTIS_ORDER, GTICS_ORDER

############################################ TESTE INSTS ###########################################

TEST_INSTS = []
PAPIER_INSTS = []
for elm in INSTS_ORDER:
	exec(f"from package.insts.{elm}.py.test_inst_{elm} import TEST_INST_{elm.upper()}, PAPIER_INST_{elm.upper()}")
	exec(f"TEST_INSTS += [TEST_INST_{elm.upper()}]")
	exec(f"PAPIER_INSTS += [PAPIER_INST_{elm.upper()}]")

########################################### TESTE SCORES ###########################################

TEST_SCORES = []
PAPIER_SCORES = []
for elm in SCORES_ORDER:
	exec(f"from package.scores.{elm}.py.test_score_{elm} import TEST_SCORE_{elm.upper()}, PAPIER_SCORE_{elm.upper()}")
	exec(f"TEST_SCORES += [TEST_SCORE_{elm.upper()}]")
	exec(f"PAPIER_SCORES += [PAPIER_SCORE_{elm.upper()}]")

############################################ TESTE OPTIS ###########################################

TEST_OPTIS = []
PAPIER_OPTIS = []
for elm in OPTIS_ORDER:
	exec(f"from package.optis.{elm}.py.test_opti_{elm} import TEST_OPTI_{elm.upper()}, PAPIER_OPTI_{elm.upper()}")
	exec(f"TEST_OPTIS += [TEST_OPTI_{elm.upper()}]")
	exec(f"PAPIER_OPTIS += [PAPIER_OPTI_{elm.upper()}]")

############################################ TESTE GTICS ###########################################

TEST_GTICS = []
PAPIER_GTICS = []
for elm in GTICS_ORDER:
	exec(f"from package.gtics.{elm}.py.test_gtic_{elm} import TEST_GTIC_{elm.upper()}, PAPIER_GTIC_{elm.upper()}")
	exec(f"TEST_GTICS += [TEST_GTIC_{elm.upper()}]")
	exec(f"PAPIER_GTICS += [PAPIER_GTIC_{elm.upper()}]")

TESTS = [TEST_INSTS, TEST_SCORES, TEST_OPTIS, TEST_GTICS]
PAPIERS = [PAPIER_INSTS, PAPIER_SCORES, PAPIER_OPTIS, PAPIER_GTICS]