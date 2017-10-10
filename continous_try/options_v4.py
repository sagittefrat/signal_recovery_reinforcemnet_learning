#inspired by: https://github.com/rlcode/reinforcement-learning/blob/a497d719e3ecdd254e6620cf4f4b9afb0524b099/2-cartpole/3-reinforce/cartpole_reinforce.py


import csv
import random 
import numpy as np
import pandas as pd
from copy import copy
import matplotlib.pyplot as plt
import networkx as nx
from scipy import optimize
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam



class Network():
	
	def __init__(self,(cluster,edges_weight_dict)):
		self.graph=cluster
		#print '\n&&&&&&&&&&&&&&& The clustering graph is: &&&&&&&&&&&&&&&&&&&\n %s' %self.graph
		self.discovered_edges={}
		self.num_nodes=len(self.graph)
		self.sampled_nodes=[]
		self.edges_weight_dict=self.get_dict_weight(edges_weight_dict)
		self.num_actions=self.number_of_actions()
		self.discovered_states_dict={}	
		self.discovered_nodes=set()
		

	''' this is to make impact of the boundry nodes, in order to make better signal guess at predict()'''
	def get_dict_weight(self,edges_weight_dict):
		'''for edge in edges_weight_dict:
			if edges_weight_dict[edge]!=1: edges_weight_dict[edge]=100'''
		return edges_weight_dict

	def number_of_actions(self):
		return len(set(self.edges_weight_dict.values()))


	def get_weight(self,edge):
		return self.edges_weight_dict[edge]

	def create_state(self,state_name):

		if state_name in self.discovered_states_dict: 
			state_features= self.discovered_states_dict[state_name]

		else:
			self.sampled_nodes.append(state_name)
			state_neighbors=[]
			#seen_nodes=[x for x in self.discovered_nodes]
			seen_nodes=[]
			for edge in self.edges_weight_dict:
				if state_name in edge:
					edges_weight=self.get_weight(edge)
					self.discovered_edges[edge]=edges_weight
					state_neighbors.append(edges_weight)
					seen_nodes.append(edge[0])
					seen_nodes.append(edge[1])

			self.discovered_nodes=set(seen_nodes+list(self.discovered_nodes))

			state_features=self.find_state_features(state_name,state_neighbors, seen_nodes)
			#state_features=(state_neighbors.count(5),state_neighbors.count(1))
			self.discovered_states_dict[state_name]=state_features
		
		#print '\n&&&&&&&&&&&&&&& sampled_nodes: &&&&&&&&&&&&&&&&&&&\n %s\n \n&&&&&&&&&&&&&&&&&&& discovered_nodes: &&&&&&&&&&&&&&&&&&&\n %s' %(self.sampled_nodes,self.discovered_nodes)
		print '\n&&&&&&&&&&&&&&& sampled_nodes: &&&&&&&&&&&&&&&&&&&\n %s  ' %(self.sampled_nodes,)
		return state_name, state_features

	# state features are: (clustering coefficient, number of neighbors with w=1, number of neighbors with w=5 )
	def find_state_features(self, state_name, state_neighbors, seen_nodes, max_w=5):
		weights_neighborhood=[]
		deg_state=len(seen_nodes)
		for v in seen_nodes:
			for u in seen_nodes:

				if state_name in (u,v): continue

				edge1=(state_name,u) if self.edges_weight_dict.has_key((state_name,u)) else (u,state_name)
				edge2=(state_name,v) if self.edges_weight_dict.has_key((state_name,v)) else (v,state_name)

				if (u,v) in self.edges_weight_dict: 
					weights_neighborhood.append((self.edges_weight_dict[(u,v)]*self.edges_weight_dict[edge1]*self.edges_weight_dict[edge2])**0.333)
		
		state_neighbors.sort()
		cc_state=sum(weights_neighborhood)/(deg_state*(deg_state-1))



		return np.array([cc_state, state_neighbors.count(1), state_neighbors.count(5)])


class Q_estimator():

	def __init__(self, state_size=3, action_size=2):

		
		# get size of state and action
		self.state_size = state_size
		self.action_size = action_size

		# lists for the states, actions and rewards
		self.states, self.actions, self.rewards = [], [], []
		


	# In Policy Gradient, Q function is not available.
	# Instead agent uses sample returns for evaluating policy
	def discount_rewards(self, rewards):
		discounted_rewards = np.zeros_like(rewards)
		running_add = 0
		for t in reversed(range(0, len(rewards))):
			running_add = running_add * self.discount_factor + rewards[t]
			discounted_rewards[t] = running_add/(t+1)
		#print 'discounted_rewards', discounted_rewards 
		return discounted_rewards


class q_nn(Q_estimator):

	
	def __init__(self, state_size, action_size):
	  
		Q_estimator.__init__(self,state_size, action_size)
		# These are hyper parameters for the Policy Gradient
		self.discount_factor = 0.9
		self.learning_rate = 0.001
		#self.hidden1 = 24
		#self.hidden2 = 24

		# create model for policy network
		self.model = self.build_model()


	# approximate policy using Neural Network
	# state is input and probability of each action is output of network
	def build_model(self):
		model = Sequential()
		model.add(Dense(self.action_size, input_dim=self.state_size, activation='softmax', kernel_initializer='glorot_uniform'))
		
		#model.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
		#model.add(Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform'))
		#model.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
		model.summary()
		# Using categorical crossentropy as a loss is a trick to easily
		# implement the policy gradient. Categorical cross entropy is defined
		# H(p, q) = sum(p_i * log(q_i)). For the action taken, a, you set 
		# p_a = advantage. q_a is the output of the policy network, which is
		# the probability of taking the action a, i.e. policy(s, a). 
		# All other p_i are zero, thus we have H(p, q) = A * log(policy(s, a))
		model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate))
		return model


	# using the output of policy network, pick action stochastically
	def get_action(self, state_name, state_features):
		state_features = np.reshape(state_features, [1, self.state_size])
		policy = self.model.predict(state_features, batch_size=1).flatten()
		#print 'policy', policy
		a = np.random.choice(self.action_size, 1, p=policy)[0]
		return a

	

	# update policy network every episode
	def update_q(self,next_state=None):

		#print 'states:', self.states
		#print 'rewards', self.rewards
		episode_length = len(self.states)

		discounted_rewards = self.discount_rewards(self.rewards)
		#print 'discounted_rewards', discounted_rewards
		
		'''discounted_rewards -= np.mean(discounted_rewards)
		if np.std(discounted_rewards)>0:
			discounted_rewards /= np.std(discounted_rewards)
		print 'discounted_rewards', discounted_rewards'''
		
		update_inputs = np.zeros((episode_length, self.state_size))
		advantages = np.zeros((episode_length, self.action_size))

		for step in range(episode_length):
			update_inputs[step] = self.states[step][1]
			advantages[step][self.actions[step]] = discounted_rewards[step]

		self.model.fit(update_inputs, advantages, epochs=1, verbose=0)
		#this is if we want to update only at the end of the epoch:
		self.states, self.actions, self.rewards = [], [], []

	# save <s, a ,r> of each step
	def append_sample(self, state_name, state_features, action, reward):
		self.states.append(state_features)
		self.rewards.append(reward)
		self.actions.append(action)
		#print ' rewards inside append sample: ', self.rewards


def make_Q(num_nodes, mode='continous', state_size=3, action_size=2):
	

	# in case the feature is between (0,1):
	if mode == 'continous':
		Q = q_nn(state_size, action_size)

	return Q
'''
This func define what is the function to do mini,aztion on :
TV of the discovered edges by the formulation: wij * | x[i]-x[j] |
'''
def l1_func(x,network,x_len):

	to_minimize=[]
	#print y[-1]
	for (node_j,node_i) in network.discovered_edges:
		#print (node_j,node_i)
		to_minimize.append((network.discovered_edges[(node_j,node_i)]**1)*abs(x[node_i]-x[node_j]))
		#to_minimize.append((network.discovered_edges[(node_j,node_i)]**1)*abs(x[node_i]-x[node_j])+x[node_i+x_len]+x[node_j+x_len])

	return np.sum(to_minimize)

	
'''
This func predicts the x[i] values of the nodes that we came across while we are traveling on the graph.
In respect to the discoverd nodes signal (as equality constrain).
The goal is to minimize the TV on those nodes.
'''
def predict(x,network,bounds):
	x_len=len(x)
	constraints = []
	y=np.zeros(2*x_len)

	y[:x_len]=x[:]
	for node in network.sampled_nodes:
		constraints.append({'type': 'eq', 'fun' : lambda x: np.array(y[node] - network.graph[node])})
	constraints_tuple = tuple(constraints)

	#sol=optimize.minimize(l1_func, x, method='SLSQP', args=(network,), constraints=constraints_tuple, tol=1e-8)
	bounds_all=[]
	for i in xrange(x_len):
		bounds_all.append(bounds)

	
	#bounds_all+=((-0.1,0.1 )for i in xrange(x_len))
	#print bounds_all

	'''for (node_j,node_i) in network.discovered_edges:
		print (node_j,node_i)'''

	#sol=optimize.minimize(l1_func, y,method='L-BFGS-B',bounds=bounds_all, args=(network,x_len)) 
	sol=optimize.minimize(l1_func, x,method='Nelder-Mead', args=(network,x_len), tol=1e-8) 
	
	#print sol.x
	return sol

def generate_graph(num_nodes=30, num_clusters=4, w=(1,5), num_nodes_inside_cluster=140, num_nodes_outside_cluster=9):
	
	edges_list=[]
	edges_weight_dict={}

	clust=[random.randint(1,num_clusters) for p in xrange(num_nodes)]

	for i in xrange(num_nodes):
		for j in xrange(i+1,num_nodes):
			if clust[i]==clust[j]:
				edge=(i,j)
				edges_list.append(edge)
				edges_weight_dict[edge]=w[1]


	random.shuffle(edges_list)
	edges_list=edges_list[:num_nodes_inside_cluster]

	for i in xrange(num_nodes_outside_cluster):
		edge=tuple(random.sample(xrange(num_nodes), 2))
		if clust[edge[0]]!=clust[edge[1]]:
			edges_weight_dict[edge]=w[0]
			edges_list.append(edge)



	with open('network_graph.csv', 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=',',
					quoting=csv.QUOTE_MINIMAL)

		#simple w[ij]={1,5}:
		for edge in edges_weight_dict:
			writer.writerow((edge[0],edge[1], edges_weight_dict[edge]))

	with open('network_clust.csv', 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=',',
					quoting=csv.QUOTE_MINIMAL)

		writer.writerow(clust)

	return (clust,edges_weight_dict)

def get_graph(network_graph='network_graph.csv',network_clust='network_clust.csv'):

	edges_weight_dict={}
	clust=[]


	if network_graph[-3:]=='csv':

		read_graph=pd.read_csv(network_graph, header=None)
		for i in xrange(len(read_graph)):
			edges_weight_dict[(read_graph.ix[i, 0],read_graph.ix[i , 1])]=read_graph.ix[i , 2]

		read_cluster=pd.read_table(network_clust,sep=',', header=None)
		for i in xrange(len(read_cluster.ix[0,:])):
			clust.append(read_cluster.ix[0,i])

	elif network_graph[-3:]=='dat':

		read_graph=pd.read_table(network_graph, sep='\s+', header=None)
		for i in xrange(len(read_graph)):
			edges_weight_dict[(read_graph.ix[i, 0]-1,read_graph.ix[i , 1]-1)]=np.round(read_graph.ix[i , 2])
		
		read_cluster=pd.read_table(network_clust, sep=',', header=None)
		for i in xrange(len(read_cluster.ix[:,1])):
			clust.append(read_cluster.ix[i,1])


	edges_list=edges_weight_dict.keys()

	G = nx.DiGraph()
	G.add_edges_from(edges_list)
	pos = nx.spring_layout(G)
	nx.draw_networkx(G)
	plt.savefig('./network.png')
	
	return (clust,edges_weight_dict)


def error(x,network):

	# calculate error by NMSE:
	sum=0
	for node in network.discovered_nodes:
		node_value=network.graph[node]
		sum+=np.square(x[node]-node_value)/np.square(node_value)	
	
	return sum

