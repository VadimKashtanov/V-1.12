PROGRAMMES_A_COMPILER = [
#	Tester le Paquet
	"test_package",									#	Tester C/Cuda
	"test_package_papier",							#	Assurer la coherance entre papier et objets en lignes=1 set=1
	
	"test_package_mdl_forward_forward2",			#	Comparer les mdl() et les fonctions pour calculer les derivee
	"test_package_mdl_1e5",							#	Comparer dS/dw et 1e5*(S(w+1e-5)-S(w))
	"test_package_mdl_1e10",						#	ddS. Et indirectement df & ddf controlent les sets et les lignes

	"test_package_1e5_1e10",						#	Compare dS, ddS avec 1e5, 1e10 avec python sur _meand et _dd_weight

	"test_package_score_1e5",						#
	"test_package_score_1e10",						#

	"gwiriadur_tutis",								#	tester tout
] + [
#	Manipulation Donnee
	"print_data",									#
	"print_line_format",							#
	"change_data_lines",							#
	"random_data",									#
] + [
#	Modeles
	"print_mdl",									#	Outil de Base pour print les models
	"test_mdl",										#	Juste print un resultat avec mdl et data
	"mdl_weight_limits",							#	Coup les poids en 2 valeurs
#	"compare_mdl_outputs",							#
] + [
#	Generation Simple Modeles
	"cli_stack_model",								#
	"join_mdls_feedforward",						#
#	Generation en Serie des Modelss
#	"agnostic_random_network1d_tensor4d",			#
] + [
#	Optmisation
	"optimize_smart",								#
#	"optimize_mdl",									#
#	"python_optimize",								#
#	"moyenne_derivee_seconde",						#
] + [
#	Visualistion
#	"test_scores",									#
#	"visualise_weight_gradient",					#
] + [
#	Comparaisons Performances et Utilisations
#	"compare_hessian_with_optis",					#
]

PROGRAMMES_QUE_PYTHON = [
#	Des idees
#	"network_gui",									# pas commencé
]


##
##	Les programmes sont les suivants
##
##		test_package						|	Tester toutes les instructions, les optimizeurs, les ...
##		test_package_python					|	Compare <inst>_backward grad's and meand's with (f(x+1e-5) - f(x))*1e5
##		test_package_python_mdl_forward 	|	Compare Toute instruction pour Mdl.mdl avec Mdl.forward en sets=1
##		compare_mdl_outputs					|	Compare Outputs of 2 models in compare_mdl_outputs/compare.py
##
##		cli_stack_model						|	Generate Feed Forward Stack Model
##		mdl_weight_limits					|	Tous les weights entre Borne0 et Borne1
##
##		optimize_mdl						|	Optimiser un model a partire d'un Data file. C'est tout.
##		python_optimize						|	Same but only python
##		test_optimize_python_mdl_functions	|	Tester la performance d'optimisation d'un fonction a travers Mdl.mdl(...) et pas forward()/backward()
##		optimize_smart						|	Interactiv Loss function
##
##		print_mdl							|	Just show mdl
##		print_data							|	Print data
##		print_line_format					|	Print data line in 2dsquare of histogram
##
##		test_mdl							|	Forward a batch with Cpu_t
##		test_scores							|	Score() on all test data batchs from out_mdl.bin
##
##		change_data_lines					|	Change (batchs,lines) with (batchs*lines/new_line,new_lines) in Data_t bin file
##		visualise_weight_gradient			|	Show a graph with Meand's of 1 weight. (With a certain precision)
##
##		random_data							|	Generate random data
##
##		join_mdls_feedforward				|	Join list of models into output
##
##		agnostic_random_network1d_tensor4d	|	Agnostic Random FeedForward network with <Ax,Ay,Az,An> tensors
##
##		moyenne_derivee_seconde				|	Moyenne des dérivées secondes
##		compare_hessian_with_optis			|	Comparer les changements entre un opti et l'hessienne
##