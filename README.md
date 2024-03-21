# Le principe résumé #

```
Avec des mots
	S est une fonction sur R+. On cherche a trouver les miniums de maniere iterative avec dS/dw et d(dS/dwi)/dwi.
	On minimise [S(f(entree, poids), sortie) pour entree, sortie dans données]
	A nous de trouver quelle S trouver aboutire a une fonction f qui satisfasse nos conditions, quelle methode d'optimisation ...
Ex:
	On part d'un 'a'. Puis on cherche un S(x)=0 a partire du maximum d'info qu'on a sur S(a)
Tangente a la courbe : S(x) ~= S(a) + S'(a)*(x - a) <=> -S(a) = S'(a)*(x-a) <=> a-S(a)/S'(a) = x
Algorithme {
	x = aléatoire()
	repeter N fois {
		x = x - S(x)/S'(x)
	}
}
```

# What to Do #

Add all programs to the ```Makefile```

Build the programs of the package with all ```order.py``` in each ```/insts``` ```/optis``` ```/scores```

```sh

	make
	rm *.o

```

```*.o``` is juste to clean and make sure everything is recompiled each time, but you can save them if you want.

## First Steps ##

If there is:
	`./test_package`, `./test_package_python`, `./test_package_python_mdl_forward`

Run them

### This package ###

If you wan to test the package with some data, build ```/my_test``` in ```/tests```.

Add ```/bin```, ```/config_files```, ```/python```, ```my_data``` and past in the last dir your data.

Save in ```/bins``` all binary files like data in ```Data_t``` format, models in ```Mdl_t``` format ...

In ```/config_files``` same file with configuration to Optimize or build models.

### Building Models ###

You can build some feed-forward networks (stack of layers) using 

```sh

	./cli_stack_model <path to the config file>

```

or juste making in ```/python``` file with ```class <My_Model>(Heritance):``` and build them with ```.bins()```

### Build Data ###

You can build in ```Data_t``` format all data using python file, or some programs to do it.

### Optimize ###

Write python file in ```/config_files``` a dictionnay with all required parameters (```optimize_mdl/compile.py```)

```sh

	./optimize_mdl <path to the config file>

```

### Test you model ###

You can make a ```.py``` file but also use ```./test_mdl```

```sh

	./test_mdl optimized_mdl.bin test_data.bin <batch> <line>

```

# Expert #

## Ajouter/Tester une instruction ##

```sh

	python3 add_inst.py Inst_name mod0,mod1 P0,P1,P2,P3,P4 Required0,Required1

```

Pas oublier le ```build_package.py```

Puis il faut evidement tester l'instruction.

### Ecrire le mdl() ###

- Puis cree dans ```tests/``` un dossier de test avec un model et des donnés qui prend cette nouvelle instruction.

- Enfin testons pour la premiere fois l'instruction (Il va afficher de la performance, donc une évolution de score).

```sh

	./test_optimize_python_mdl_functions mon_dossier_de_test

````

### Ecrire le forward() et le backward() ###

Juste en Python, puis tester que ça correspond bien avec mdl() et que backward() a un sens :

```sh

	./test_package_python
	./test_package_python_mdl_forward

```

Puis Utiliser le meme dossier `mon_dossier_de_test`, et tester l'optimisation python avec forward(), backward()

```sh

	./python_optimize mon_dossier_de_test/config_files/instruction_to_build

````

### Enfin Ecrire le C/Cuda ###

Et tester tout

```sh
	
	clear; make; rm *.o
	./test_package
	./optimize_mdl mon_dossier_de_test/config_files/instruction_to_build

```

### Eventuellement utiliser des outils graphique pour visualiser et affiner des choses ###

Tester le 279 parametre avec une precision de 4 (1-2 sera largement suffisant, sauf si on veut vraiment taper un grosse moyenne)

C'est globalement pas tres utile car c'est ultra variable, et a un sens sur des tout petits models.

```sh

	./visualise_weight_gradient mon_dossier_de_test/bins/mdl.bin 279 4

```