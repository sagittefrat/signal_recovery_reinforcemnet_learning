#SLP

import numpy as np

def sparse_label_prop (D, sampling_set, sampling_vals, MAX_ITER, epsilon):
	''' 
	% Predicts labels starting form an input subset
	% Optimized version of the SLP algorithm which uses nornalization by sum of Cols/rows of D
	%   Implements the SLP algorithm by Jung et. al.
	%   Input: 
	%          D (N_edges x N_vertices)
	%          sampling_set:   indices of samples
	%          sampling_vals:  samples values
	%          epsilon:        tolerance value as stopping criteria
	%          MAX_ITER:       maximum number of iterations to run the alg.
	%   Output: 
	%          hatx:           new signal values for all nodes

	'''
	# Initial graph info
	NE = D.shape[0]     # Number of edges
	NV = D.shape[1]    # Number of vertices
	ones_vect = np.ones(NE,1)
	zeros_vect = np.zeros(NE,1)


	# Other variables
	lamda = 1.0		# Controls weight between primal and dual variables
	alpha = 2.0

	def pow2alpha(x):
		return x**(alpha-2)

	def powalpha(x):
		return x**alpha

	aux1 = pow2alpha(abs(D))
	aux2 = powalpha(abs(D))

	T = scipy.sparse.spdiags(1.0/(lamda * (np.sum(aux1,1).conj().transpose())) , 0, NV, NV)
	E = scipy.sparse.spdiags(lamda/np.sum(aux2,2), 0, NE, NE)

	ED = scipy.sparse(E.dot(D))
	TD = scipy.sparse(T .dot(D.conj().transpose()))

	% Initialize some variables
	z = np.zeros(NV,1)
	xk = np.zeros(NV,1)
	hatx = np.zeros(NV,1)
	y = np.zeros(NE,1)

	#get the value of sampled nodes:
	j=0
	for i in sampling_set:
		xk[j] = sampling_vals[i]
		j+=1

	k = 1
	while 1:
		
		# Step 1
		signal = y + (ED.dot(z))   
		#y = sign(signal) .* max(horzcat(abs(signal)-epsilon, zeros_vect),[],2);


		for i in xrange(NE):

			y[i] = (1.0 / np.max(np.concatenate(abs(signal)[i,:],1)) * signal
		
		# Step 2
		r = xk - (TD.dot(y))
		
		# Step 3 
		xk1 = r
		j=0
		for i in sampling_set:
			xk[j] = sampling_vals[i]
			j+=1
		
		# Step 4 
		z = 2.0 * xk1 - xk
		
		# stopping criteria
		#if((k > 1) & (xk1-xk < epsilon))
		if k == MAX_ITER : #| (1/(k-1) * hatx) - (1/(k) * (hatx + xk1)) < epsilon)
			return 1/(MAX_ITER-1) * hatx
		
		# Step 5
		hatx = hatx + xk1
		
		xk = xk1
		k = k+1


	hatx = 1/(k-1) * hatx

	return hatx




