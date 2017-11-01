import numpy as np 
import matplotlib as mpl
import pdb 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
 
# Test data 
n = 15 

xarray = np.linspace(-5 - n,5 + n,n)
yarray = np.linspace(-5 - n,5 + n,n)
n *= n 
xarray, yarray = np.meshgrid(xarray, yarray)
# xarray = xarray.flatten()
# yarray = yarray.flatten()
Xtest = np.vstack([xarray.flatten() , yarray.flatten() ]).T
print '--Xtest shape:', Xtest.shape
def rbf_kernel_D_vectorized(mat1, mat2, sigma):
	"""A vectorized rbf kernel
	:input   mat1: N x D  where D is the dimension of each entry data, N is the number of data
	:input  sigma: sigmaeter of the rbf kernel
	:output  N x D kernel value
	: reference: https://stats.stackexchange.com/questions/239008/rbf-kernel-algorithm-python/239073 """
	trnorms1 = np.mat([(v * v.T)[0, 0] for v in mat1]).T
	trnorms2 = np.mat([(v * v.T)[0, 0] for v in mat2]).T

	k1 = trnorms1 * np.mat(np.ones((mat2.shape[0], 1), dtype=np.float64)).T

	k2 = np.mat(np.ones((mat1.shape[0], 1), dtype=np.float64)) * trnorms2.T

	k = k1 + k2
 
	k -= 2 * np.mat(mat1 * mat2.T)

	k *= - 1./(2 * np.power(sigma, 2))

	return np.array(np.exp(k)) 


sigma =  1

 
K_ss = rbf_kernel_D_vectorized(np.matrix(Xtest), np.matrix(Xtest), sigma)

print '--shape K_ss:', K_ss.shape 
 

# Get cholesky decomposition (square root) of the
# covariance matrix
L = np.linalg.cholesky(K_ss + 1e-15*np.eye(n)) 

# Sample 3 sets of standard normals for our test points,
# multiply them by the square root of the covariance matrix
f_prior = np.dot(L, np.random.normal(size=(n,3)))
 
 

 
# fig = plt.figure()
# ax = fig.gca(projection='3d')

# zarray = f_prior[:,1]/np.max(f_prior[:,1])
# zarray = zarray.reshape(xarray.shape)
# surf = ax.plot_surface(xarray, yarray, zarray ,
#                        linewidth=0, antialiased=False)
# plt.show()

def fun(x, y):
  return np.sin(x) + 0*np.abs(y)

Ntrain = 10 
xarray_train = np.linspace(-3 - Ntrain,3 + Ntrain, Ntrain)
yarray_train = np.linspace(-3 - Ntrain,3 + Ntrain, Ntrain)
 
xarray_train, yarray_train = np.meshgrid(xarray_train, yarray_train)
ytrain  = np.array([fun(x ,y ) for x, y in zip(np.ravel(xarray_train), np.ravel(yarray_train))])
# Noiseless training data
Xtrain = np.vstack([xarray_train.flatten() , yarray_train.flatten() ]).T
 

# Apply the kernel function to our training points
K = rbf_kernel_D_vectorized(np.matrix(Xtrain), np.matrix(Xtrain), sigma)
L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain)))

 
# Compute the mean at our test points.
K_s = rbf_kernel_D_vectorized(np.matrix(Xtrain), np.matrix(Xtest), sigma)
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


# Compute error 
ytest_gt  = np.array([fun(x ,y ) for x, y in zip(np.ravel(xarray), np.ravel(yarray))])
RMSE_test = np.mean((ytest_gt-mu)**2)
print '>> Resule RMSE: %.4f'%RMSE_test 
# Visualize training and testing data 
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
zarray = ytrain  
zarray = zarray.reshape(xarray_train.shape)
surf = ax.plot_surface(xarray_train, yarray_train, zarray ,
                       linewidth=0, antialiased=False)
ax.set_title('Training') 



ax = fig.add_subplot(122, projection='3d') 
ax.set_title('Testing, RMSE = %.3f'%RMSE_test)

zarray = mu 
zarray = zarray.reshape(xarray.shape)
surf = ax.plot_surface(xarray, yarray, zarray , linewidth=0, antialiased=False, label='Pred', color=(0.9, 0.3,0.3)) #, cmap='CMRmap')
 
zarray = ytest_gt.reshape(xarray.shape)
surf = ax.plot_surface(xarray, yarray, zarray , linewidth=0, antialiased=False, color=(0.3, 0.9, 0.7), label='Gt') #, cmap='cool')
 
plt.show()