import numpy as np
import random, pdb 
import util 

from scipy import sparse

from sklearn.cluster import MiniBatchKMeans
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler

# rbf_kernel: K(x, y) = exp(-gamma ||x-y||^2)


class SparseHilbertMap(object):
	"""Hilbert map using the sparse feature."""
	def __init__(self, centers, gamma=1.0, cutoff=0.001, use_rkhs=True):
		"""Creates a new sparse hilbert map.

		:param centers the locations at which to create sparse kernels
		:param gamma the gamma parameter of the RBF kernel
		:param cutoff value below which kernel values are set to 0
		:param use_rkhs whether or not to use reproducing kernel hilbert spaces
		"""
		self.centers = centers 
		self.gamma = gamma
		self.cutoff = cutoff 
		self.use_rkhs = use_rkhs
		self.batch_size = 100  
		self.classifier = SGDClassifier(\
			loss="log",\
			penalty="elasticnet",\
			alpha=0.0001,\
			class_weight=None,\
			l1_ratio=0.80)

	def add(self, data, labels):
		"""Updates the classifier with new data.

		:param data the raw data points to add
		:param labels the labels associated with the data
		"""
		# pdb.set_trace()
		# util.scatter2D(data)
		if self.use_rkhs:
			process_data = self.rkhs_data(data)
			if process_data.shape[0] < 100:
				self.classifier.fit(process_data, labels)
			else:
				offset = 0 
				while offset < process_data.shape[0]:
					if offset+self.batch_size >= process_data.shape[0]:
						print('> Next batch out of index!')
						self.classifier.partial_fit( \
									process_data[offset:],\
					        labels[offset:], \
					        classes=[0, 1])
					else:			
						self.classifier.partial_fit( \
									process_data[offset:offset+self.batch_size],\
					        labels[offset:offset+self.batch_size], \
					        classes=[0, 1])

					offset += self.batch_size 
		else:
			if len(data) < 100:
				kernel = rbf_kernel(data, self.centers, self.gamma)
				kernel = sparse.csr_matrix(kernel * (kernel > self.cutoff))
				self.classifier.fit(kernel, labels)
			else:
			  offset = 0
			  while offset < len(data):
			    # Compute kernel matrix and sparsify it
			    kernel = rbf_kernel(data[offset:offset+self.batch_size], self.centers, self.gamma)
			    kernel = sparse.csr_matrix(kernel * (kernel > self.cutoff))

			    # Update the classifier using the kernel matrix
			    self.classifier.partial_fit(
			            kernel,
			            labels[offset:offset+self.batch_size],
			            classes=[0, 1]
			      
			    )
			    offset += self.batch_size

	def rkhs_data(self, data):
		"""Return features obtained using the RKHS for the given data.

		:param data the data entries for which to create RKHS content
		:return RKHS based features for the given input data
		"""		
		avg_features = None
		# Compute average feauters for each input point
		for entry in data:
		    mean_map_sample_coords = np.random.multivariate_normal(
		            entry,
		            [(0.2, 0.0), (0.0, 0.2)],
		            10
		    )
		    kernel = rbf_kernel(mean_map_sample_coords, self.centers, self.gamma)
		    kernel = kernel.mean(axis=0)
		    kernel = sparse.csr_matrix(kernel * (kernel > self.cutoff))

		    if avg_features is None:
		        avg_features = kernel
		    else:
		        avg_features = sparse.vstack((avg_features, kernel))
		return avg_features
		


	def classify(self, query):
		"""Returns the probabilistic prediction for the given data points.

		:param query the data points to perform predictions on
		:return predicted values for the given inputs
		"""
		if self.use_rkhs:
			kernel = self.rkhs_data(query)
		else:
			kernel = rbf_kernel(query, self.centers, self.gamma)
			kernel = sparse.csr_matrix(kernel*(kernel > self.cutoff))
		return self.classifier.predict_proba(kernel)
