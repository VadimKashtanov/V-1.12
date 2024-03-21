INSTS_ORDER = [
#	Premieres Instructions
#	"activation",				#df&ddf
	"dot1d",					#df&ddf
#	"dot2d",					#df
#	"kconvl2d",					#df
#	"pool2dmax",				#df
#	"pool2daverage",			#df
#	"softmax",					#df
] + [
#	Instructions Classiques
#	"lstm2d",					#df
#	"dropout",					#df
#	"dotconvl1d",				#df
#	"hadamard1d",				#df
] + [
#	Reunion d'instructions classiques
#	"dot1dsoftmax",				#df
] + [
#	Spetiales sur instructions classiques
#	"dot2dsinglebias",			#df
#	"dot2drecurent",			#df
#	"simplebias1d",				#df
#	"simpledot2dweight",		#df
] + [
#	Manipulation techniques
#	"sum",						#df
#	"mul",						#df
#	"time1dcopy",				#df
#	"mulscalar",				#df
] + [
#	Instructions non conventionnelles
#	"gaussfiltre2d",			#df
#	"dotgaussfiltre2d",			#df
#	"dotgaussfiltre3d",			#df
] + [
#	Bac a sable
#	"forcedotgaussfiltres2daexpbxc",	#df
#	"atanlink1d",				#df
] + [
#	Bac a sable BTC-USDT
#	"meme1d",					#df&ddf ( simple memoire avec 1 boucle)
]