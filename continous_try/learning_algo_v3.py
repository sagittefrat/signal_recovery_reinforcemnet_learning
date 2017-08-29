# coding=utf-8

'''
This is the code for the state features represented as: ( CC(v), number of neighbrs with w=5, number of neighbrs with w=1 )
with sampling 12 nodes out of 30 of 4 clusters, usually we discover all edges, and then, the empirical error is around 11.5
we need to integrate with the minimizing TV costumized algorithm
'''

from options_v3 import *


# create the problem graph:
#network = Network(generate_graph(30,4))

# use the graph already created:
network = Network (get_graph())

def learn():


	# define the learning parameters:	
	#alpha=0.9
	#alpha_decay=0.9
	#exploration=0.9
	#exploration_decay=0.9
	#epsilon=0.1
	#epsilon_decay=1
	#beta=0.01
	#policy_type='epsilon_greedy'
	#policy_type='softmax'
	#discount_factor=0.9
	num_steps=7

	#gamma=0.9

	# the problem size:
	num_nodes=network.num_nodes
	num_actions=network.num_actions
	
	# create Q table for discrete state space and discrete actions:
	

	Q=make_Q(num_nodes, mode='continous', state_size=3, action_size=num_actions+1)
	
	
	# create initial state - a random node from the graph:
	state_name, state_features=network.create_state(int(random.random()*num_nodes))
	print 'initial state: %s' %state_name
	
	# start the graph walk - choosing the best next action by following the RL best policy:
	for step in xrange(num_steps):
		print '\n&&&&&&&&&&&&&&&&&&& step: %s &&&&&&&&&&&&&&&&&&&' %step
	

		action=Q.get_action(state_name, state_features)

		'''if exploration<random.random():
			next_state=network.create_state(int(random.random()*num_nodes))
			action=num_actions
			print '\n&&&&&&&&&&&&&&&&&&&  random_action!, next state: %s &&&&&&&&&&&&&&&&&&& ' %next_state[0]
		
		else:'''
		discovered_edges=copy(network.discovered_edges)

		edge_to_sample=[]
		for edge in discovered_edges:
			if state_name in edge:
				if network.edges_weight_dict[edge]==action or 5-network.edges_weight_dict[edge]==action:
					edge_to_sample.append(edge)
					
		if edge_to_sample==[]: 
			#action=num_actions
			next_state=network.create_state(int(random.random()*num_nodes))
		else:
			#print edge_to_sample
			random.shuffle(edge_to_sample)
			edge_to=edge_to_sample[0]
			#print edge_to
			next_state=network.create_state(edge_to[1] if edge_to[0]==state_name else edge_to[0])


		# guess the initial x[i] of the discovered nodes:
		x=np.random.rand(num_nodes)
		possible_values=[]
		for node in network.sampled_nodes:
			possible_values.append(network.graph[node])
			x[node]=network.graph[node]
		possible_values.sort()
		
		#update X values to be 
		for node in network.discovered_nodes:
			if node not in network.sampled_nodes:
				x[node]=x[node]*(possible_values[-1]-possible_values[0])+possible_values[0]


		# solve the TV optimization problem with the initial guess: 
		sol= predict(x,network)
		tv=sol.fun

		Q.append_sample(state_name,state_features, action, tv)
		Q.update_q(next_state)

		print '\n&&&&&&&&&&&&&&&&&&& TV: %s &&&&&&&&&&&&&&&&&&&' %tv	

		print  'state name: %s, state features: %s action: %s, next state" %s '  %(state_name, state_features, action, next_state[0])
		
		state_name,state_features=next_state
		
		#alpha*=alpha_decay
		#exploration*=exploration_decay	
		#raw_input()

	####### end of epiosde:
	#Q.model.save_weights("./save_model/signal_recovery_model.h5")
	empirical_error=error(sol.x,network)
	print 'lenght of discovered nodes:', len(network.discovered_nodes) 
	print 'clust',network.graph
	print 'x', x
	print 'empirical error:',empirical_error
	print 'sum(clust-x):',np.sum(abs(network.graph-sol.x))



if __name__ == '__main__':
	learn()









