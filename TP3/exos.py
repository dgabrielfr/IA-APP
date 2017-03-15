import numpy as np
import math

n = 10 ** 6
data = 10 ** 9 + np.random.uniform(0,1,n)

def moyenne (donnees,nb):
	return float(sum(donnees))/float(nb)

test_moyenne = moyenne (data,n)
print "La moyenne est de : ", test_moyenne

def variance(donnees,nb):
	m = moyenne (donnees,nb)
	v = 0
	for i in donnees:
		v = v + math.pow((i-m),2)
	return float(v)/float(nb-1)

test_variance = variance (data, n)
print "La variance est de :", test_variance

def moyenne_Welford (donnees,nb):
	m = float(donnees[0])
	for i in range(1,nb):
		m = m + (float(donnees[i])-m)/float(i)
	return m

test_moyenne_Welford = moyenne_Welford (data,n)
print "La moyenne de Welford est de :", test_moyenne_Welford

def variance_Welford (donnees,nb):
	v = 0.0
	m = float(donnees[0])
	for i in range(1,nb):
		v = v + (float(donnees[i])-m)*(float(donnees[i])-(m + (float(donnees[i])-m)/float(i)))
		m = m + (float(donnees[i])-m)/float(i)
	return v/(nb-1)

test_variance_Welford = variance_Welford (data,n)
print "La variance de Welford est de :", test_variance_Welford


