# coding=utf-8

'''
This is the code for the state features represented as: ( CC(v), min weight of neighbours, max weight of neighbours )
on the LFR graph from n-lasso paper
we need to integrate with the minimizing TV costumized algorithm
'''

from options_v3 import *


# create the problem graph:
#network = Network(generate_graph(num_nodes=2000, num_clusters=30, w=(1,5), num_nodes_inside_cluster=1000, num_nodes_outside_cluster=100))
#network = Network(generate_graph(30,4))

# use the graph already created:
#network = Network (get_graph('../SLPCode/LFR/c30_4/network.dat', '../SLPCode/LFR/c30_4/community.dat'))
network1 = Network (get_graph())
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
	num_nodes=network1.num_nodes
	num_actions=network1.num_actions
	
	# create Q table for discrete state space and discrete actions:
	

	Q=make_Q(num_nodes, mode='continous', state_size=3, action_size=num_actions+1)
	
	for i in xrange(11):
		network = Network (get_graph())
	
		# create initial state - a random node from the graph:
		state_name, state_features=network.create_state(int(random.random()*num_nodes))
		print 'initial state: %s' %state_name
		
		# start the graph walk - choosing the best next action by following the RL policy:
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
					if network.edges_weight_dict[edge]==action:
						edge_to_sample.append(edge)
						
			if edge_to_sample==[]: 
				action=num_actions
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

			Q.append_sample(state_name,state_features, action, -tv)
			Q.update_q(next_state)

			print '\n&&&&&&&&&&&&&&&&&&& TV: %s &&&&&&&&&&&&&&&&&&&' %tv	

			print  'state name: %s, state features: %s action: %s, next state" %s '  %(state_name, state_features, action, next_state[0])
			
			state_name,state_features=next_state

			print 'number of discovered nodes:', len(network.discovered_nodes) 

			
			#alpha*=alpha_decay
			#exploration*=exploration_decay	
			#raw_input()

		####### end of epiosde:
		#Q.model.save_weights("./save_model/signal_recovery_model.h5")
		empirical_error=error(sol.x,network)
		print 'empirical error:',empirical_error
		raw_input()
	print 'clust',network.graph
	print 'x', x
	#print 'empirical error:',empirical_error
	print 'sum(clust-x):',np.sum(abs(network.graph-sol.x))



if __name__ == '__main__':
	learn()









