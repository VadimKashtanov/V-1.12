#pragma once

//	Pour package/ et head/, dans des #define la quantité d'objs
#include "package/meta.cuh"

//	Ce .cuh inclue tous les .cuh du head/
#include "kernel/head/testpackage.cuh"

//	Chaque obj a un .cuh ou toutes les (*)[] sont stoqué
#include "package/insts/insts.cuh"
#include "package/scores/scores.cuh"
#include "package/optis/optis.cuh"
#include "package/gtics/gtics.cuh"

//	Arrays are declared in headers and writed in package/src/*.cu