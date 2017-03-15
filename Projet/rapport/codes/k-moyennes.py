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

def distance(x,y):
	dist = 0
	for i in range(0,len(x)):
		dist += pow(x[i]-y[i],2)
	return math.sqrt(dist)


def k_moyennes (dico_app,dico_test,nb_app,nb_test,nb_periode):

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

		nb_eval = [0]*10
		mod_moyenne = modele

		for donnee in data_apprentissage:

			l_dist = [0]*10

			for j in range(0,10):
				l_dist[j] = distance(donnee[1],modele[j])

			ind_min = 0
			dist_min = l_dist[0]

			for d in range(1,10):
				if l_dist[d] < dist_min:
					ind_min = d
					dist_min = l_dist[d]

			nb_eval[ind_min] += 1

			for n in range (0,1024):
				mod_moyenne[ind_min][n] = (mod_moyenne[ind_min][n]*nb_eval[ind_min] + donnee[1][n])/(nb_eval[ind_min]+1)


		modele = mod_moyenne

		nb_erreur_test = 0
		l_mal_reco = [0]*10

		for donnee in data_test:

			l_dist = [0]*10

			for j in range(0,10):
				l_dist[j] = distance(donnee[1],modele[j])

			ind_min = 0
			dist_min = l_dist[0]

			for d in range(1,10):
				if l_dist[d] < dist_min:
					ind_min = d
					dist_min = l_dist[d]

			if ind_min != donnee[0]:
				nb_erreur_test = nb_erreur_test + 1
				l_mal_reco[donnee[0]] += 1

		taux_erreur_test = float(nb_erreur_test)/float(len(data_test))

		print "Le taux d'erreur des tests à l epoque", i, "est de :" ,taux_erreur_test
		print "Le nombre d'erreur par type d'image est de :" , l_mal_reco
		
		l_mal_reco_periode.append(l_mal_reco)
		
        	# Extraction des infos par classe (donc colonne)
	
	nb_erreur_classe = []
	
	for i in range(0,10) :
            nb_erreur_classe.append(zip(*l_mal_reco_periode)[i])
		
	plt.xlabel("Epoque")
	plt.ylabel("Nombre d'image mal reconnues")
	plt.title("Perceptron multi classe")
	for i in range(0,10) :
            plt.plot(nb_erreur_classe[i], label=dic_label["label_names"][i])
        plt.legend()
	plt.grid()
	# plt.show()
	plt.savefig("k-moyennes.png")
	plt.clf()
	

k_moyennes(dic_apprentissage,dic_test,10000,10000,30)
