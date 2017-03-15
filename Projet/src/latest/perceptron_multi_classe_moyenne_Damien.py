#coding: utf-8

import matplotlib.pyplot as plt
import numpy
import math
import random

def unpickle(file):
	import cPickle
	fo = open(file, 'rb')
	dic = cPickle.load(fo)
	fo.close()
	return dic


dic_apprentissage = unpickle('cifar-10-batches-py/data_batch_1')
dic_test = unpickle('cifar-10-batches-py/test_batch')

dic_label = unpickle('cifar-10-batches-py/batches.meta')

# print dic_label["label_names"][0]


def data_reader(dico,k):
	data = []

	for i in range(0,k):
		l_valeur_pixel = []
		for j in range(0,1024):
				valeur_pixel = math.sqrt(pow(dico["data"][i][j],2) + pow(dico["data"][i][j+1024],2) + pow(dico["data"][i][j+2048],2))
				l_valeur_pixel.append(valeur_pixel)
		data.append([dico["labels"][i],l_valeur_pixel])
	return data

def prod_scal(x,y):
	res = 0
	for i, j in zip(x,y):
		res = res +  i * j	
	return res

def add_scal(x,y):
	res = [0]*len(x)
	for i in range(0,len(x)):
		res[i] = x[i] + y[i]	
	return res

def mul_scal(x,y):
	res = [0]*len(y)
	for i in range(0,len(y)):
		res[i] = x*y[i]	
	return res

def perceptron_multi_classe (dico_app,dico_test,nb_app,nb_test,nb_periode):

	data_apprentissage = data_reader(dico_app,nb_app)
	data_test = data_reader(dico_test,nb_test)

	modele = [[0]*1024]*10

	b = 0
	l_b = [0]*10
	c = 0

	while b < 10:
		if l_b[data_apprentissage[c][0]] == 0:
			modele[data_apprentissage[c][0]] = data_apprentissage[c][1]
			l_b[data_apprentissage[c][0]] = 1
			b += 1
		c += 1


        l_mal_reco_periode = []
        
        for i in range(1,nb_periode+1):
            
            
		# tester sans le suffle
		# random.shuffle(data_apprentissage)
		nb_erreur_apprentissage = 0
		mod_moyenne = modele
		ite = 1

		for donnee in data_apprentissage:

			l_score = [0]*10

			for j in range(0,10):

				score_perceptron = prod_scal(donnee[1],modele[j])
				l_score[j] = score_perceptron

			ind_max = 0
			score_max = l_score[0]

			for s in range(1,10):
				if l_score[s] > score_max:
					ind_max = s
					score_max = l_score[s]

			if ind_max != donnee[0]:
				nb_erreur_apprentissage = nb_erreur_apprentissage + 1
				modele[donnee[0]] = add_scal(modele[donnee[0]],donnee[1])
# tester non moyenné
#			for m in range(0,10):
#				for n in range(0,1024):
#					mod_moyenne[m][n] = (mod_moyenne[m][n] * ite + modele[m][n])/(ite+1)
			ite += 1
# tester non moyenné
#		modele = mod_moyenne
		
		nb_erreur_test = 0
		l_mal_reco = [0]*10

		for donnee in data_test:

			l_score = [0]*10

			for j in range(0,10):

				score_perceptron = prod_scal(donnee[1],modele[j])
				l_score[j] = score_perceptron

			ind_max = 0
			score_max = l_score[0]

			for s in range(1,10):
				if l_score[s] > score_max:
					ind_max = s
					score_max = l_score[s]

			if ind_max != donnee[0]:
				nb_erreur_test = nb_erreur_test + 1
				l_mal_reco[donnee[0]] += 1
				
		taux_erreur_apprentissage = float(nb_erreur_apprentissage)/(float(len(data_apprentissage)))
		taux_erreur_test = float(nb_erreur_test)/float(len(data_test))

		print "Le taux d'erreur d'apprentissage à l'epoque", i, "est de :" ,taux_erreur_apprentissage
		print "Le taux d'erreur des tests à l epoque", i, "est de :" ,taux_erreur_test
		print "Le nombre d'erreur par type d'image est de :" , l_mal_reco
                
                l_mal_reco_periode.append(l_mal_reco)
	
	# Extraction des infos par classe (donc colonne)
	
	nb_erreur_classe = []
	
	for i in range(0,10) :
            nb_erreur_classe.append(zip(*l_mal_reco_periode)[i])
	
	print nb_erreur_classe
	
	plt.xlabel("Epoque")
	plt.ylabel("Nombre d'image mal reconnues")
	plt.title("Perceptron multi classe")
	for i in range(0,10) :
            plt.plot(nb_erreur_classe[i], label=dic_label["label_names"][i])
        plt.legend()
	plt.grid()
	# plt.show()
	plt.savefig("Perceptron_multiclasse_non_moyenne.png")
	plt.clf()

perceptron_multi_classe(dic_apprentissage,dic_test,10000,10000,100)
