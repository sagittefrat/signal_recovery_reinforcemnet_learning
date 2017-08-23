
import csv
import random 
import numpy as np
import pandas as pd
from copy import copy
import matplotlib.pyplot as plt
import networkx as nx
from scipy import optimize



class Network():
	
	def __init__(self,(cluster,edges_weight_dict)):
		self.graph=cluster
		print '\n&&&&&&&&&&&&&&& The clustering graph is: &&&&&&&&&&&&&&&&&&&\n %s' %self.graph
		self.discovered_edges={}
		self.num_nodes=len(self.graph)
		self.sampled_nodes=[]
		self.edges_weight_dict=edges_weight_dict
		self.num_actions=self.number_of_actions()
		self.discovered_states_dict={}	
		self.discovered_nodes=set()
		


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
			seen_nodes=[]
			for edge in self.edges_weight_dict:
				if state_name in edge:
					edges_weight=self.get_weight(edge)
					self.discovered_edges[edge]=edges_weight
					state_neighbors.append(edges_weight)
					seen_nodes.append(edge[0])
					seen_nodes.append(edge[1])

			self.discovered_nodes=set(seen_nodes)


			state_features=(state_neighbors.count(5),state_neighbors.count(1))
			self.discovered_states_dict[state_name]=state_features
		
		print '\n&&&&&&&&&&&&&&& sampled_nodes: &&&&&&&&&&&&&&&&&&&\n %s\n \n&&&&&&&&&&&&&&&&&&& discovered_nodes: &&&&&&&&&&&&&&&&&&&\n %s' %(self.sampled_nodes,self.discovered_nodes)
		return state_name, state_features
'''
This func define what is the function to do mini,aztion on :
TV of the discovered edges by the formulation: wij * | x[i]-x[j] |
'''
def l1_func(x,network):

	to_minimize=[]

	for (node_j,node_i) in network.discovered_edges:
		to_minimize.append(network.discovered_edges[(node_j,node_i)]*abs(x[node_i]-x[node_j]))
	
	return np.sum(to_minimize)

	
'''
This func predicts the x[i] values of the nodes that we came across while we are traveling on the graph.
In respect to the discoverd nodes signal (as equality constrain).
The goal is to minimize the TV on those nodes.
'''
def predict(x,network):
	
	constraints = []

	for node in network.sampled_nodes:
		constraints.append({'type': 'eq', 'fun' : lambda x: np.array(x[node] - network.graph[node])})
	constraints_tuple = tuple(constraints)

	sol=optimize.minimize(l1_func, x,method='SLSQP', args=(network,),constraints=constraints_tuple, tol=1e-4)
	
	print sol.x
	return sol

def generate_graph(num_nodes=30,num_clusters=4, w=(1,5),num_nodes_inside_cluster=140,num_nodes_outside_cluster=9):
	
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



	G = nx.DiGraph()
	G.add_edges_from(edges_list)
	pos = nx.spring_layout(G)
	nx.draw_networkx(G)
	plt.savefig('./network.png')
	#plt.show()


	with open('network.csv', 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=',',
					quoting=csv.QUOTE_MINIMAL)

		#simple w[ij]={1,5}:
		for e in edges_weight_dict:
			writer.writerow((e[0],e[1], edges_weight_dict[e]))

	return (clust,edges_weight_dict)



def select_action(Q,s,policy_type,beta,exploration_rate,policy):

	action_probs = policy(s)
	if (1 - exploration_rate) < np.random.uniform(0, 1):
		a= np.random.choice(np.arange(len(action_probs)))

	else:
		# Select the first action in this episode

		if policy_type == 'softmax':
			a = select_a_with_softmax(Q[s,:], beta=beta)
		elif policy_type == 'epsilon_greedy':
			a = np.random.choice(np.arange(len(action_probs)), p=action_probs)
		elif policy_type == 'random':
			a = np.argmin(Q[s,:] + np.random.randn(1,len(action_probs)))
		else:
			raise ValueError("Invalid policy_type: {}".format(policy_type))
		
		print 'a', a
		return a


def make_epsilon_greedy_policy(Q, epsilon, num_actions):
	"""
	Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
	
	Args:
		Q_estimator: An estimator that returns q values for a given state
		epsilon: The probability to select a random action . float between 0 and 1.
		nA: Number of actions in the environment.
	
	Returns:
		A function that takes the observation as an argument and returns
		the probabilities for each action in the form of a numpy array of length action_space_n.
	
	"""
	def policy_fn(observation):
		A = np.ones(num_actions, dtype=float) * epsilon / num_actions
		q_values = Q[observation,:]
		print 'q_values', q_values
		best_action = np.argmin(q_values)
		print 'best_action',best_action
		print 'A',A
		A[best_action] += (1.0 - epsilon)
		return A

	return policy_fn



def softmax(Q, beta=1.0):
	
	assert beta >= 0.0
	q_tilde = Q - np.min(Q)
	factors = np.exp(beta * q_tilde)

	return factors / np.sum(factors)

def select_a_with_softmax(q_values, beta=1.0):

	prob_a = softmax(q_values, beta=beta)
	cumsum_a = np.cumsum(prob_a)

	return np.where(np.random.rand() < cumsum_a)[0][0]

def select_a_with_epsilon_greedy(s, Q, epsilon=0.1):

	a = np.argmin(Q[s, :])
	if np.random.rand() < epsilon:
		a = np.random.randint(Q.shape[1])

	return a


def error(x,network):

	# calculate error by NMSE:
	sum=0
	for node in network.discovered_nodes:
		node_value=network.graph[node]
		sum+=np.square(x[node]-node_value)/np.square(node_value)	
	
	return sum

