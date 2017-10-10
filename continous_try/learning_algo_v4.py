# coding=utf-8

'''
This is the code for the state features represented as: ( CC(v), min weight of neighbours, max weight of neighbours )
on the LFR graph from n-lasso paper
'''

from options_v4 import *


# create the problem graph:
#network1 = Network(generate_graph(num_nodes=200, num_clusters=7, w=(1,5), num_nodes_inside_cluster=100, num_nodes_outside_cluster=10))
#network1 = Network(generate_graph(30,4))

# use the graph already created:
#network = Network (get_graph('../SLPCode/LFR/c30_4/network.dat', '../SLPCode/LFR/c30_4/community.dat'))
network1 = Network (get_graph())
def learn():
	num_episodes=20
	num_steps=7

	# the problem size:
	num_nodes=network1.num_nodes
	num_actions=network1.num_actions
	
	
	Q=make_Q(num_nodes, mode='continous', state_size=3, action_size=num_actions+1)
	
	for epiosde in xrange(num_episodes):
		network = Network (get_graph())
		prev_tv=0
		# create initial state - a random node from the graph:
		state_name, state_features=network.create_state(int(random.random()*num_nodes))
		#print '\n&&&&&&&&&&&&&&&&&&& episode: %s &&&&&&&&&&&&&&&&&&&' %episode
		
		# start the graph walk - choosing the best next action by following the RL policy:
		for step in xrange(num_steps):
			#print '\n&&&&&&&&&&&&&&&&&&& step: %s &&&&&&&&&&&&&&&&&&&' %step
		

			action=Q.get_action(state_name, state_features)
			#print 'action chosen:',action,
			discovered_edges=copy(network.discovered_edges)

			action=action%5
			edge_to_sample=[]
			dif_action_to_sample=[]
			dif_action=1-action
			for edge in discovered_edges:
				if state_name in edge:
					# if ther's an adge then the weight can be only 0/1
					if network.edges_weight_dict[edge]%5==action :
						edge_to_sample.append(edge) 
						#print 'found edge to sample', edge

					else :
						dif_action_to_sample.append(edge) 
					
						
			if action==2: next_state=network.create_state(int(random.random()*num_nodes))
			# IF THERE IS NO MAtching weight then choose the other action 0/1 w.p 0.5
			elif edge_to_sample==[]:
				if random.random()<0.33:
					random.shuffle(dif_action_to_sample)
					edge_to=dif_action_to_sample[0]
					next_state=network.create_state(edge_to[1] if edge_to[0]==state_name else edge_to[0])

				else:
					#action=num_actions
					next_state=network.create_state(int(random.random()*num_nodes))
					
			
			else:
				random.shuffle(edge_to_sample)
				edge_to=edge_to_sample[0]
				next_state=network.create_state(edge_to[1] if edge_to[0]==state_name else edge_to[0])

		
				
			#print 'action taken, next state:',action, next_state[0]
			#print 'edge_to_sample:', edge_to_sample

			# guess the initial x[i] of the discovered nodes:
			x=np.random.rand(num_nodes)
			possible_values=[]
			for node in network.sampled_nodes:
				possible_values.append(network.graph[node])
				x[node]=network.graph[node]
			possible_values.sort()
			(min_val,max_val)=(possible_values[-1],possible_values[0])
			#print '(min_val,max_val)', (min_val,max_val)
			
			#update X values to be 
			for node in network.discovered_nodes:
				if node not in network.sampled_nodes:
					x[node]=x[node]*(min_val-max_val)+max_val


			# solve the TV optimization problem with the initial guess: 
			sol = predict(x,network, (min_val,max_val))
			#print '\n&&&&&&&&&&&&&&&&&&& sol.fun: %s &&&&&&&&&&&&&&&&&&&' %sol.fun
			to_minimize=[]
			for (node_j,node_i) in network.discovered_edges:
				to_minimize.append((network.discovered_edges[(node_j,node_i)]**1)*abs(x[node_i]-x[node_j]))
			
			tv=np.sum(to_minimize)#-prev_tv
			prev_tv=np.sum(to_minimize)


			
			#print '\n&&&&&&&&&&&&&&&&&&& TV: %s &&&&&&&&&&&&&&&&&&&' %tv
			#tv = prev_tv-sol.fun
			#prev_tv=sol.fun

			Q.append_sample(state_name,state_features, action, 1/(tv+0.000000001))
			#Q.update_q(next_state)


			#print '\n&&&&&&&&&&&&&&&&&&& TV: %s &&&&&&&&&&&&&&&&&&&' %tv	

			#print  'state name: %s, state features: %s action: %s, next state" %s '  %(state_name, state_features, action, next_state[0])
			
			state_name,state_features=next_state

			print 'number of discovered nodes:', len(network.discovered_nodes) 
			
			#print 'clust',network.graph
			#print 'x', sol.x
			#raw_input()
		Q.update_q(next_state)
		empirical_error=error(sol.x,network)
		print 'empirical error:',empirical_error
		print 'clust',network.graph
		print 'x', sol.x
		
		#raw_input()
	


if __name__ == '__main__':
	learn()









