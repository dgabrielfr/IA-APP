#coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sklearn.svm

def unpickle(file):
	import cPickle
	fo = open(file, 'rb')
	dic = cPickle.load(fo)
	fo.close()
	return dic


dic_apprentissage = unpickle('cifar-10-batches-py/data_batch_1')
dic_test = unpickle('cifar-10-batches-py/test_batch')
dic_label = unpickle('cifar-10-batches-py/batches.meta')

def to_4_patchs (dico,nb_image,l_classe):

	patchs = []
	i = 0
	limite = 0

	while limite < nb_image:

		if dico["labels"][i] in l_classe:

			data_image = []
			l_data_image = []
			patch_i = []

			for j in range(0,1024):
				data_image.append([dico["data"][i][j],dico["data"][i][j+1024],dico["data"][i][j+2048]])


			for r in range(0,32):
				l_temp = []
				for t in range(0,32):
					l_temp.append(data_image[t+r*32])
				l_data_image.append(l_temp)

			

			l_temp = []
			for q in range(0,16):
				l_temp.append(l_data_image[q][16:])
			patch_i.append(l_temp)

			l_temp = []
			for q in range(0,16):
				l_temp.append(l_data_image[q][:16])
			patch_i.append(l_temp)

			l_temp = []
			for q in range(0,16):
				l_temp.append(l_data_image[q+16][16:])
			patch_i.append(l_temp)

			l_temp = []
			for q in range(0,16):
				l_temp.append(l_data_image[q+16][:16])
			patch_i.append(l_temp)

			for num_patch in range(0,4):

				somme_patch = 0
				moyenne = 0
				sd = 0

				for r in range(0,16):
					for t in range(0,16):
						somme_patch += sum(patch_i[num_patch][r][t])

				moyenne = float(somme_patch)/float(768)

				for r in range(0,16):
					for t in range(0,16):
						for p in range(0,3):
							sd += pow((patch_i[num_patch][r][t][p] - moyenne),2)
				standard_deviation = math.sqrt(float(sd)/float(768))

				for r in range(0,16):
					for t in range(0,16):
						for p in range(0,3):
							if standard_deviation != 0:
								patch_i[num_patch][r][t][p] = float(patch_i[num_patch][r][t][p]-moyenne)/float(standard_deviation)
								
					

			patchs.append([dico["labels"][i],patch_i])
			limite += 1

		i += 1

	return patchs

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

def normalize(lst):
    s = sum(lst)
    if s == 0:
	return [0]*len(lst)
    else:
   	 return map(lambda x: float(x)/s, lst)

def traitement_image_apprentissage (dico_app,nb_app,nb_periode,l_classe):

	data_apprentissage = to_4_patchs(dico_app,nb_app,l_classe)
	nb_classe = len(l_classe)

	modele = [[[[[0]*3]*32]*32]*4]*nb_classe
	random.shuffle(data_apprentissage)

	b = 0
	l_b = [0]*nb_classe
	c = 0

	while b < nb_classe:
		if l_b[data_apprentissage[c][0]] == 0:
			modele[data_apprentissage[c][0]] = data_apprentissage[c][1]
			l_b[data_apprentissage[c][0]] = 1
			b += 1
		c += 1


	for i in range(1,nb_periode+1):

		mod_moyenne = modele
		nb_eval = [[0]*4]*10

		for donnee in data_apprentissage:

			for num_patch in range(0,4):

				l_dist = [0]*10

				for r in range (0,16):

					for t in range(0,16):

						for p in range(0,3):

							for m in range(0,nb_classe):
								l_dist[m] += pow(int(donnee[1][num_patch][r][t][p])-int(modele[m][num_patch][r][t][p]),2)

			
				for m in range(0,nb_classe):
					l_dist[m] = math.sqrt(l_dist[m])
				ind_min = 0
				dist_min = l_dist[0]

				for d in range(1,nb_classe):
					if l_dist[d] < dist_min:
						ind_min = d
						dist_min = l_dist[d]

				nb_eval[ind_min][num_patch] += 1

				for r in range (0,16):

					for t in range(0,16):

						for p in range(0,3):

							mod_moyenne[ind_min][num_patch][r][t][p] = (mod_moyenne[ind_min][num_patch][r][t][p]*nb_eval[ind_min][num_patch] + donnee[1][num_patch][r][t][p])/(nb_eval[ind_min][num_patch]+1)

		modele = mod_moyenne


	images_traitees = []

	for donnee in data_apprentissage:

		image_representation = []

		for num_patch in range(0,4):

			l_dist = [0]*nb_classe

			for r in range (0,16):

				for t in range(0,16):

					for p in range(0,3):

						for m in range(0,nb_classe):
							l_dist[m] += pow(int(donnee[1][num_patch][r][t][p])-int(modele[m][num_patch][r][t][p]),2)


			for m in range(0,nb_classe):
					l_dist[m] = math.sqrt(l_dist[m])
			ind_min = 0
			dist_min = l_dist[0]

			for d in range(1,nb_classe):
				if l_dist[d] < dist_min:
					ind_min = d
					dist_min = l_dist[d]

			for ind in range(0,nb_classe):
				if ind == ind_min:
					image_representation.append(1)
				else:
					image_representation.append(0)

		images_traitees.append([donnee[0],image_representation])

	return images_traitees,modele

def traitement_image_test (dico_test,nb_test,modele,l_classe):

	data_test = to_4_patchs(dico_test,nb_test,l_classe)
	images_traitees = []
	nb_classe = len(l_classe)

	for donnee in data_test:

		image_representation = []

		for num_patch in range(0,4):

			l_dist = [0]*nb_classe

			for r in range (0,16):

				for t in range(0,16):

					for p in range(0,3):

						for m in range(0,nb_classe):
							l_dist[m] += pow(int(donnee[1][num_patch][r][t][p])-int(modele[m][num_patch][r][t][p]),2)

			for m in range(0,nb_classe):
					l_dist[m] = math.sqrt(l_dist[m])
			ind_min = 0
			dist_min = l_dist[0]

			for d in range(1,nb_classe):
				if l_dist[d] < dist_min:
					ind_min = d
					dist_min = l_dist[d]

			for ind in range(0,nb_classe):
				if ind == ind_min:
					image_representation.append(1)
				else:
					image_representation.append(0)

		images_traitees.append([donnee[0],image_representation])

	return images_traitees

def classification_image(dico_app,dico_test,nb_app,nb_test,nb_periode_traitement,l_classe):

	data_apprentissage,modele = traitement_image_apprentissage (dico_app,nb_app,nb_periode_traitement,l_classe)
	data_test = traitement_image_test (dico_test,nb_test,modele,l_classe)

	Y = np.array([donnee[0] for donnee in data_apprentissage])
	X = np.array([donnee[1] for donnee in data_apprentissage])

	clf = sklearn.svm.SVC()
	clf.fit(X, Y)
	sklearn.svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    	decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    	max_iter=-1, probability=False, random_state=None, shrinking=True,
    	tol=0.001, verbose=False)

	nb_erreur_test = 0
	l_mal_reco = [0]*10

	for donnee in data_test:
		if clf.predict([donnee[1]]) != donnee[0]:
				nb_erreur_test = nb_erreur_test + 1
				l_mal_reco[donnee[0]] += 1
				
	taux_erreur_test = float(nb_erreur_test)/float(len(data_test))
	
	print "Le taux d'erreur des tests est de :", taux_erreur_test
	print "Le nombre d'erreur par type d'image est de :", l_mal_reco


classification_image(dic_apprentissage,dic_test,1000,100,5,[0,1,2,3,4,5,6,7,8,9])
