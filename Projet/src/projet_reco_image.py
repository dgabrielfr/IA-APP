#coding: utf-8
# identation : aligner (sous Kate)
# ctrl+D : comment
# ctrl+shit+D : decomment

## TODO : faire un perceptron multi classe (un perceptron pour chaque classe)
##	bonne classe = celle dont le score est max
##	renvoyer le score (et modele?) pour chaque perceptron

import matplotlib.pyplot as plt
import numpy

def unpickle(file):
	import cPickle
	fo = open(file, 'rb')
	dic = cPickle.load(fo)
	fo.close()
	return dic
    
def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise


dic_apprentissage = unpickle('data_batch_1')
dic_test = unpickle('test_batch')

dic_label = unpickle('batches.meta')

# print dic["labels"][0]
# print label

# Plot de l'image avec ses différents canaux
# TODO : mettre dans une fonction avec un indice
# ATTENTION : ne pas oublier le paramètre 'F' dans le reshape
# ATTENTION-2 : il faut aussi transposer après le reshape sinon l'image est pivotée de 90°
# Le premier indice varie en premier (comme en Fortran) et pas comme en C/C++
# TODO : Découper en plusieurs fichiers ?


for indice in range(1,100):

    fig = plt.figure()
    fig.suptitle(dic_label["label_names"][dic_test["labels"][indice]], fontsize=14)

    image = plt.subplot(2,2,1)
    image.set_title("Image")
    image.imshow(dic_test["data"][indice][0:1024*3].reshape(32,32,3,order='F').transpose(1,0,2))

    #image_r = plt.subplot(2,2,2)
    #image_r.set_title("Composante R")
    #image_r.imshow(dic_apprentissage["data"][0][0:1024].reshape(32,32, order='F').transpose(1,0,2))

    #image_g = plt.subplot(2,2,3)
    #image_g.set_title("Composante G")
    #image_g.imshow(dic_apprentissage["data"][0][1024:1024*2].reshape(32,32, order='F').transpose(1,0,2))

    #image_b = plt.subplot(2,2,4)
    #image_b.set_title("Composante B")
    #image_b.imshow(dic_apprentissage["data"][0][1024*2:1024*3].reshape(32,32, order='F').transpose(1,0,2))

    print ("Ceci est un : " + dic_label["label_names"][dic_apprentissage["labels"][indice]])

    nom = "image_"
    nom += str(indice)

    mkdir_p('./images/')

    plt.savefig('./images/' + nom + '.png')

    fig.clf()
    #plt.show()

    
def data_reader(dico):
	data = []
	for i in range(0,50):
		l_label = []
		l_vect = []
		for j in range(0,1024):
			vect_pixel = [dico["data"][i][j],dico["data"][i][j+1024],dico["data"][i][j+2048]]
			l_vect.append(vect_pixel)
		for k in range(0,10):
			label = 0
			if k == dico["labels"][i]: 
				label = 1 
			else:
				label = -1
			l_label.append(label)
		data.append([l_label,l_vect])
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

#def perceptron_data(dico_app,dico_test,n,nom_plot):

	#x = []*n
	#y_test = []*n
	#y_apprentissage = []*n

	#data_apprentissage = data_reader(dico_app)
	
	#data_test = data_reader(dico_test)

	#modele = [0]*len(data_apprentissage[0][1])

	#for i in range(1,n+1):
		#nb_erreur_apprentissage = 0

		#for donnee in data_apprentissage:
			#l_valeur_pixel = []

			#for j in range(0,len(donnee[1])):
				#valeur_pixel = donnee[1][j][0] + donnee[1][j][1]*256 + donnee[1][j][2]*256*256
				#l_valeur_pixel.append(valeur_pixel)
			#for k in range(0,10):
				#if donnee[0][k]*prod_scal(l_valeur_pixel,modele) <= 0:
					#modele = add_scal(modele,mul_scal(donnee[0][k],l_valeur_pixel))
					#nb_erreur_apprentissage = nb_erreur_apprentissage + 1

		#nb_erreur_test = 0
		#for donnee in data_test:
			#l_valeur_pixel = []

			#for j in range(0,len(donnee[1])):
				#valeur_pixel = donnee[1][j][0] + donnee[1][j][1]*256 + donnee[1][j][2]*256*256
				#l_valeur_pixel.append(valeur_pixel)
			#for k in range(0,10):
				#print(prod_scal(l_valeur_pixel,modele))
				#if donnee[0][k]*prod_scal(l_valeur_pixel,modele) <= 0:
					#nb_erreur_test = nb_erreur_test + 1

		#taux_erreur_apprentissage = float(nb_erreur_apprentissage)/(float(len(data_apprentissage))*10.0)
		#taux_erreur_test = float(nb_erreur_test)/float(len(data_test)*10.0)

		#x.append(i)
		#y_test.append(taux_erreur_test)
		#y_apprentissage.append(taux_erreur_apprentissage)

		##print "Le taux d'erreur d'apprentissage à l'epoque", i, "est de :" ,taux_erreur_apprentissage
		##print "Le taux d'erreur des tests à l epoque", i, "est de :" ,taux_erreur_test

	#plt.xlabel("Epoque")
	#plt.ylabel("Taux d'erreur")
	#plt.title("Perceptron")
	#apprentissage, = plt.plot(x,y_apprentissage, label = "Apprentissage")
	#test, = plt.plot(x,y_test, label = "Test")
	#plt.legend(handles=[apprentissage, test])
	#plt.grid()
	#plt.savefig(nom_plot)
	#plt.clf()


#perceptron_data(dic_apprentissage,dic_test,10,"methode_naive_apprentissage.png")
