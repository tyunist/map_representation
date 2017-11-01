import numpy as np 
import pdb 
import matplotlib.pyplot as plt 

# Test data 
n = 50 
Xtest = np.linspace(-5,5,n).reshape(-1,1) # n x 1 

def rbf_kernel_1D_vectorized(a, b, param):
	sqdist = np.sum(a**2, 1).reshape(-1,1) + np.sum(b**2, 1)  - 2*np.dot(a, b.T)

	return np.exp(-.5*(1/param) * sqdist)


param = 0.1 

K_ss = rbf_kernel_1D_vectorized(Xtest, Xtest, param)

# Get cholesky decomposition (square root) of the
# covariance matrix

L = np.linalg.cholesky(K_ss + 1e-15*np.eye(n)) 

# Sample 3 sets of standard normals for our test points,
# multiply them by the square root of the covariance matrix
f_prior = np.dot(L, np.random.normal(size=(n,3)))

 

# plt.plot(Xtest, f_prior)
# plt.axis([-5, 5,  -3, 3])
# plt.title('Three sampels from the GP prior')

# plt.show()


 

# Noiseless training data
Xtrain = np.array([-4, -3, -2, -1, 1]).reshape(5,1)
ytrain = np.sin(Xtrain)

# Apply the kernel function to our training points
K = rbf_kernel_1D_vectorized(Xtrain, Xtrain, param)
L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain)))

# Compute the mean at our test points.
K_s = rbf_kernel_1D_vectorized(Xtrain, Xtest, param)
Lk = np.linalg.solve(L, K_s)  

# mu = K(Xtest, X) dot inv[K(X, X) + var*I]* ytrain 
# Lk =   inv(L) dot K_s 
# Lk.T =  K_s.T dot (inv(L)).T 
# np.linalg.solve(L, ytrain)) = inv(L) dot y_train 
# np.dot(Lk.T, np.linalg.solve(L, ytrain)) =  K_s.T dot (inv(L)).T  dot inv(L) dot y_train = K_s.T dot inv(K) dot y_train which is 
# consistent with what is introduced in the textbook! 
# ( L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain))) so L dot L.T = K so inv(K) = (inv(L)).T  dot inv(L))  
mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((n,))

# Compute the standard deviation so we can plot it
s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
stdv = np.sqrt(s2)
# Draw samples from the posterior at our test points.
# Output covariance:
# K_ss - np.dot(Lk.T, Lk)
L = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,3)))

plt.plot(Xtrain, ytrain, 'bs', ms=8)
# plt.plot(Xtest, f_post)
plt.gca().fill_between(Xtest.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
plt.plot(Xtest, mu, 'r--', lw=2)
plt.axis([-5, 5, -3, 3])
plt.title('Three samples from the GP posterior')
plt.show()


