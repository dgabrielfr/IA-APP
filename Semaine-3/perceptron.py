import numpy as np


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
