import numpy as np
import math
import random

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


def data_reader(filename):
    to_binary = {"?": 3, "y": 2, "n": 1}
    labels = {"democrat": 1, "republican": -1}

    data = []
    for line in open(filename, "r"):
        line = line.strip()

        label = int(labels[line.split(",")[0]])
        observation = np.array([to_binary[obs] for obs in line.split(",")[1:]] + [1])
        data.append((label, observation))

    return data


def spam_reader(filename):
    to_binary = {1: 1, 0: -1}
    data = []
    for line in open(filename, "r"):
        line = line.strip()
        label = to_binary[int(line.split(",")[-1])]
        observation = [float(obs) for obs in line.split(",")[:-1] + [1.0]]

        data.append((label, np.array(observation)))
        
    return data
