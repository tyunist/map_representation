import numpy as np 
import pdb 
def fun(X):
	# X input = N x D where N is number of data
	# pdb.set_trace()
	assert isinstance(X, np.ndarray)
	if len(X.shape) < 2: 
		X = np.expand_dims(X, 0)
	sigma = np.eye(X.shape[1])
	mean = np.ones(X.shape[1])
	epsilon = 2
	return np.array([np.exp(-1./(2*epsilon**2)*np.dot(np.dot(np.reshape(v-mean, [1, -1]), \
	            sigma), np.reshape(v-mean, [1, -1]).T) )[0,0] for v in X])


X = np.array([[2, 2], [1,1]])
print np.ones(X.shape[1])
out = fun(X)
print out 