# coding=utf-8

from options import *


# create the problem graph:
network=Network(generate_graph(30,4))


def learn():


	# define the learning parameters:	
	alpha=0.9
	alpha_decay=0.9
	exploration=0.9
	exploration_decay=0.9
	epsilon=0.1
	#epsilon_decay=1
	beta=0.01
	policy_type='epsilon_greedy'
	#policy_type='softmax'
	discount_factor=0.9
	num_steps=7
	gamma=0.9

	# the problem size:
	num_nodes=network.num_nodes
	num_actions=network.num_actions
	
	# create Q table for discrete state space and discrete actions:
	average_mistake=(3*num_nodes*5)/2
	Q=np.zeros((num_nodes,num_actions))+average_mistake

	
	# create initial state - a random node from the graph:
	state_name, state_features=network.create_state(int(random.random()*num_nodes))

	#if no FA:
	state_features=state_name
	
	# create the initial probability:
	policy = make_epsilon_greedy_policy(
			Q, epsilon , num_actions)
	#action=select_action(Q,state_features,policy_type,beta,epsilon,policy)
	print 'initial state: %s initial action: %s' %(state_name, 'bla')
	
	# start the graph walk - choosing the best next action by following the RL best policy:
	for step in xrange(num_steps):
		print '\n&&&&&&&&&&&&&&&&&&& step: %s &&&&&&&&&&&&&&&&&&&' %step


		

		action=select_action(Q,state_features,policy_type,beta,epsilon*0.9,policy)

		if exploration<random.random():
			next_state=network.create_state(int(random.random()*num_nodes))
			print '\n&&&&&&&&&&&&&&&&&&&  random_action!, next state: %s &&&&&&&&&&&&&&&&&&& ' %next_state[0]
		
		else:
			discovered_edges=copy(network.discovered_edges)

			edge_to_sample=None
			for edge in discovered_edges:
				if state_name in edge:
					if network.edges_weight_dict[edge]==action or 5-network.edges_weight_dict[edge]==action:
						edge_to_sample=edge
						
			if edge_to_sample==None: 
				next_state=network.create_state(int(random.random()*num_nodes))
			else:
				next_state=network.create_state(edge[1] if edge[0]==state_name else edge[0])


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
		print '\n&&&&&&&&&&&&&&&&&&& TV: %s &&&&&&&&&&&&&&&&&&&' %tv


		next_state_name=next_state[0]
		# Update the function approximator using our target
		Q[state_name,action]=(1-alpha)*Q[state_name,action]+alpha*(tv+gamma*np.argmin(Q[next_state_name,:]))

		
		state_name,state_features=next_state
		#if no FA:
		state_features=state_name
		
		#action=next_action
		alpha*=alpha_decay
		exploration*=exploration_decay	
		#print 'Q',Q
		raw_input()

	####### end of epiosde:
	empirical_error=error(sol.x,network)
	print 'lenght of discovered nodes:', len(network.discovered_nodes) 
	print 'clust',network.graph
	print 'x', x
	print 'empirical error:',empirical_error
	print 'sum(clust-x):',np.sum(network.graph-sol.x)



if __name__ == '__main__':
	learn()









