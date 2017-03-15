import random
import numpy as np
import math

class BernoulliArm:
	def __init__(self, p):
 		self.p = p
	def draw(self):
 		return 0.0 if random.random() > self.p else 1.0
            
            
            
            

def discret_rand(epsilon):
	if random.random() > epsilon:
		return 0
	else:
		return 1

def maxi (liste):
	tab_max = []
	val_max = 0
	for value in liste:
		if value > val_max:
			val_max = value
 	for i in range(0,len(liste)):
		if liste[i] == val_max:
			tab_max.append(i)
	ind_max = random.randrange(0,len(tab_max))
	return tab_max[ind_max]
    
def softmax (liste,temp):
        somme_reward = 0
        
        for i in liste:
             somme_reward += math.exp(float(i)/float(temp))   
             
        l_prob = []
        for i in range(0,len(liste)):
                l_prob.append( math.exp(float(liste[i])/float(temp))/float(somme_reward) )
        
        trouvee = 0
        ind = 0
        while trouvee == 0:
              if random.random() <= l_prob[ind]:
                    trouvee = 1
              else:
                    ind += 1
        return ind
        

class EpsilonGreedy:

# constructor
# epsilon (float): tradeofff exploration/exploitation
	def __init__(self, epsilon):
		self.epsilon = epsilon
# re-initialize the algorithm in order to run a new simulation
# n_arms (int): number of arms
	def initialize(self, n_arms):
		self.n_arms = n_arms
		self.dico = [0]*self.n_arms
		self.nb_change = [0]*self.n_arms
		self.l_reward = []
# return a index of the chosen decision
	def select_arm(self,temp):
		if discret_rand(self.epsilon) == 0:
			return random.randrange(0,self.n_arms)
		else:
			return softmax (self.dico,temp)
			 
# update knowledge
# chosen_arm (int): the decision that has been made
# reward (float): the obtained reward
	def update(self, chosen_arm, reward):
		self.nb_change[chosen_arm] += 1
		self.l_reward.append(reward)
		n = self.nb_change[chosen_arm]
		self.dico[chosen_arm] = ((n-1) * self.dico[chosen_arm])/n + reward/n

	def moyenne(self):
		self.res_moyenne = np.mean(self.l_reward)
		return self.res_moyenne
	

def test_algorithm(algo, means, num_sims, horizon):
# init. all decisions
	arms = [BernoulliArm(mu) for mu in means]
	rewards = []
	for sim in range(num_sims):
		algo.initialize(len(arms))
	for t in range(horizon):
                temp = 1.0/(float(t)+0.0000001)
 		chosen_arm = algo.select_arm(temp)
		reward = arms[chosen_arm].draw()
		algo.update(chosen_arm, reward)
		rewards.append(reward)
	moyenne = algo.moyenne()
	return moyenne # np.array(rewards).reshape((num_sims, horizon))

for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:

	algo = EpsilonGreedy(0.1)

	moyenne = test_algorithm (algo,[0.1, 0.1, 0.1, 0.1, 0.9],100,250)

	print "i = ",i,": moyenne = ", moyenne

