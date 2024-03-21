#
#	Gcc tools :
#		gdb, 		valgind
#		cuda-gdb
#		
#	Je peux utiliser `nvprof`
#		sudo nvprof ./progr arg0 arg1
#
#	cuda-gdb ./<program>
#	valgrind --track-origins=yes ./<package>           (eventuellement avec --leak-check=full, mais les arguments sont toujours avant le programe)
#
#
#	D'habitude je fais clear; make; rm *.o
#	Il faut faire attention a pas ecrire rm * .o
#

#device (debbug les __global__ kernel)
	#DEBBUG = -G -O0 # -Mcuda=debug
#host (debbug les fonctions classiques sur le CPU)
	DEBBUG = -g -O0 #-Mcuda=debug
#optimized out
	#DEBBUG = -O0

#			-I$(0) est le package
ARGS = $(DEBBUG) -lm -I. --compiler-options -Wall
#rdc=true <=> relocatable-device-code pour avoire des fonction __device__ par exemple dans plusieurs units. Car nvcc compile tout en inline.
#les kernels ne "call" pas de fonctions. Elles sont toutes inline car le .text de l'assembleur kernel (un truc equivalent pour les cartes nvidia)
#n'a pas de turing compatible avec .text et .data ensemble. On launch des "bloque" .text donc les fonctions __device__ doivent etre directement dedans
#    Mais il y a cette erreur
#nvlink error   : Multiple definition of 'const_mem' in 'activation_backward_th11.o', first defined in 'activation.o'

#all: compare_hessian_with_optis moyenne_derivee_seconde mdl_weight_limits agnostic_random_network1d_tensor4d join_mdls_feedforward random_data test_scores optimize_smart test_optimize_python_mdl_functions cli_stack_model test_package test_package_python test_package_python_mdl_forward optimize_mdl print_mdl print_data test_mdl print_line_format compare_mdl_outputs change_data_lines visualise_weight_gradient python_optimize
all: tous_les_programmes

*.o:
	@ printf "[\033[35;1;41m***\033[0m] =========== Configuration Avec Python =================\n"
	@ #Ici j'authorise l'executement direct de ces programs de script
	@ #chmod +x tools/terminal.py docmentation/programs tools/fast_train.sh test_all
	chmod +x bac
	@ #ici je construit (grace aux order.py) les fichier .cu qui listent toutes les fonctions/consts du package
	python3 -m build_package
	@ printf "[\033[35;1;41m***\033[0m] ================= KERNEL && PACKAGE ===================\n"
	nvcc -c $(ARGS) $(shell find package/gtics -type f -name "*.cu") $(shell find kernel -type f -name "*.cu") $(shell find package/insts -type f -name "*.cu") $(shell find package/optis -type f -name "*.cu") $(shell find package/scores -type f -name "*.cu")

tous_les_programmes: *.o
	@ printf "[\033[35;1;41m***\033[0m] ============== ### TOUS LES PROGRAMMES ### =================\n"
	python3 compile_programs.py $(ARGS)

#(depuis tres longtemps)
#Un jours je voulais lier tous les *.o
#ld -relocatable *.o -c __merge_kernel_package.o
#Pour ne pas suppr __merge_kernel_package.o
#mv __merge_kernel_package.o __merge_kernel_package.o_
#rm *.o
#mv __merge_kernel_package.o_ __merge_kernel_package.o
