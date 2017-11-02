# Reference:
# Compute grad of log likelihood GP: https://github.com/dfm/george/blob/9c920c4d7b5335d980318408372cf6b929789599/george/basic.py#L1

# Vectorized version of RBF kernel: https://stats.stackexchange.com/questions/239008/rbf-kernel-algorithm-python/239073


import argparse 
import copy 
import math 
import matplotlib.pyplot as plt 
import numpy as np 

import sys, pdb
import GP_map  

import util, os, shutil 

# online drawing 
#plt.ion()



def simple_grid(data, fname, resolution=0.1):
  poses = data['poses']
  scans = data['scans']
  xlim, ylim = util.bounding_box(poses, 40.0)
  x_count = int(math.ceil((xlim[1] - xlim[0]) / resolution))
  y_count = int(math.ceil((ylim[1] - ylim[0]) / resolution))
  map = np.zeros([y_count, x_count]) 
  plt.figure()
  for data, label in util.data_generator(poses, scans):
    #pdb.set_trace()  
    for i in range(len(label)):
      x = int((data[i][0] - xlim[0])/(xlim[1] - xlim[0])*(x_count-1))
      y = int((data[i][1] - ylim[0])/(ylim[1] - ylim[0])*(y_count-1))
      occupied = label[i] 
      map[y, x] = occupied
    print '<x, y>:', x, y, 'limit:', x_count, y_count 

  plt.imshow(map, cmap='gray')
  plt.title('Grid Occupancy Map,' + fname.split('.')[0]) 
  plt.savefig(fname)
  plt.show()
 
def generate_GP_map(model, resolution, limits, fname, verbose=True, percent_map=100):
    """Generates a grid map by querying the model at cell locations.

    :param model the hilbert map model to use
    :param resolution the resolution of the produced grid map
    :param limits the limits of the grid map
    :param fname the name of the file in which to store the final grid map
    :param verbose print progress if True
    """
    # Determine query point locations
    x_count = int(math.ceil((limits[1] - limits[0]) / resolution))
    y_count = int(math.ceil((limits[3] - limits[2]) / resolution))
    sample_coords = []
    for x in range(x_count):
        for y in range(y_count):
            sample_coords.append((limits[0] + x*resolution, limits[2] + y*resolution))

    # Obtain predictions in a batch fashion
    predictions = []
    offset = 0
    batch_size = 1000
    # pdb.set_trace()

    # Query the model 
    L_xx = model['L']
    Xtrain = model['X']
    sigma = model['sigma']
    v0 = model['v0']
    ytrain = model['y']
 
    while offset < len(sample_coords):
			Xtest = sample_coords[offset:offset+batch_size] 
			K_ss = GP_map.rbf_kernel_D_vectorized(np.matrix(Xtest), np.matrix(Xtest), sigma, v0)
			K_xs = GP_map.rbf_kernel_D_vectorized(np.matrix(Xtrain), np.matrix(Xtest), sigma, v0)
			L_xs = np.linalg.solve(L_xx, K_xs)
			mu_test = np.dot(L_xs.T, np.linalg.solve(L_xx, ytrain)).reshape((batch_size,))

			predictions.extend(mu_test)

			# s2 = np.diag(K_ss) - np.sum(L_xs**2, axis=0)
			# stdv = np.sqrt(s2)

			if verbose:
			    sys.stdout.write("\rQuerying model: {: 6.2f}%".format(offset / float(len(sample_coords)) * 100))
			    sys.stdout.flush()
			offset += batch_size
    if verbose:
        print("")
    predictions = np.array(predictions)
  

    # Turn predictions into a matrix for visualization
    mat = predictions.reshape(x_count, y_count)
    plt.clf()
    plt.title("Occupancy map %d %%"%percent_map)
    plt.imshow(mat.transpose()[::-1, :])
    plt.colorbar()
    plt.savefig(fname)
    plt.show()
    plt.pause(0.05)

def train_GP(data, components, sigma, v0, distance_cutoff, map_resolution=None, online_map=False, max_iter=1):
  """Trains a GP map model  

  :param data the dataset to train on
  :param components the number of components to use
  :param gamma the gamma value to use in the RBF kernel
  :param distance_cutoff the value below which values are set to 0
  :return hilbert map model trained on the given data
  """
  poses = data['poses']
  scans = data['scans']
  # pdb.set_trace()

  # Limits in metric space based on poses with a 10m buffer zone
  xlim, ylim = util.bounding_box(poses, 10.0)

  # Sampling locations distributed in a even grid over the area 
  centers = util.sampling_coordinates(xlim, ylim, math.sqrt(components))



  # train the model with the data 
  count = 0 
  old_percent_map = 0 
  n_iter = 0 
  n_data = 0 

  for data, label in util.data_generator(poses, scans):
    n_iter += 1 
    print '\n------------- Iter %d ---------------'%n_iter
    # print '...'
    # print 'data', data, 'label', label 
    label = label.reshape([-1, 1])
    n_ = data.shape[0]
    print '--data shape:', data.shape 
    print '--label shape:', label.shape  
    if n_iter == 1:
      n_data = n_ 
      X_total = data 
      Y_total = label 
      continue 
    # Aggregate data 
    n_data += n_ 
    X_total = np.vstack([X_total, data])
    Y_total = np.vstack([Y_total, label])
    print '--Total data:', Y_total.shape[0]
    percent_map = count/float(len(poses)) * 100

    sys.stdout.write("\rTraining model:{: 6.2f}%".format(percent_map))
    sys.stdout.flush()
    count += 1 
    if n_iter >= max_iter:
    	break 


  # Compute kernel   
  K = GP_map.rbf_kernel_D_vectorized(np.matrix(X_total), np.matrix(X_total), sigma, v0)
  print '\n--shape of K:', K.shape 

  # Get Cholesky decomposition 
  L = np.linalg.cholesky(K + 1e-5*np.eye(n_data))
 
  model = {}
  model['sigma'] = sigma 
  model['K'] = K 
  model['L'] = L 
  model['X'] = X_total
  model['v0'] = v0
  model['y'] = Y_total
  return model 

   

def train_GP_incremental(data, components, sigma, v0, distance_cutoff, map_resolution=None, online_map=False):
  """Trains a GP map model  

  :param data the dataset to train on
  :param components the number of components to use
  :param gamma the gamma value to use in the RBF kernel
  :param distance_cutoff the value below which values are set to 0
  :return hilbert map model trained on the given data
  """
  poses = data['poses']
  scans = data['scans']
  # pdb.set_trace()

  # Limits in metric space based on poses with a 10m buffer zone
  xlim, ylim = util.bounding_box(poses, 10.0)

  # Sampling locations distributed in a even grid over the area 
  centers = util.sampling_coordinates(xlim, ylim, math.sqrt(components))



  # train the model with the data 
  count = 0 
  old_percent_map = 0 
  n_iter = 0 
  n_data = 0 

  for data, label in util.data_generator(poses, scans):
    n_iter += 1 
    print '\n------------- Iter %d ---------------'%n_iter
    # print '...'
    # print 'data', data, 'label', label 
    label = label.reshape([-1, 1])
    print '--data shape:', data.shape 
    print '--label shape:', label.shape  

    # Compute kernel of current data 
    n_ = data.shape[0]
    K_ = GP_map.rbf_kernel_D_vectorized(np.matrix(data), np.matrix(data), sigma, v0)
    print '--shape of K_:', K_.shape 

    # Get Cholesky decomposition 
    L_ = np.linalg.cholesky(K_ + 1e-5*np.eye(n_))

    if n_iter == 1:
      L_total = L_ 
      K_total = K_ 
      n_data = n_ 
      X_total = data 
      Y_total = label 
      continue 

    K_x_star = GP_map.rbf_kernel_D_vectorized(np.matrix(X_total), np.matrix(data), sigma, v0)

    K_tmp = np.zeros([n_data + n_, n_data + n_])
    K_tmp[0:n_data, 0:n_data] = K_total 
    K_tmp[0:n_data, n_data: n_data + n_] = K_x_star 
    K_tmp[n_data: n_data + n_, 0:n_data] = K_x_star.T
    K_tmp[n_data: n_data + n_, n_data:n_data + n_] = K_

    K_total = K_tmp 

    L_total = np.linalg.cholesky(K_total + 1e-5*np.eye(n_data + n_)) 


    # Eventually, aggregate data 
    n_data += n_ 
    X_total = np.vstack([X_total, data])
    Y_total = np.vstack([Y_total, label])
    print '--Total data:', Y_total.shape[0]
    percent_map = count/float(len(poses)) * 100

    sys.stdout.write("\rTraining model:{: 6.2f}%".format(percent_map))
    sys.stdout.flush()
    sys.stdout.write("\rold_percent_map:{: 6.2f}%".format(old_percent_map))
    sys.stdout.flush()
    print('\n...current %%: ', percent_map)
    # if (percent_map > old_percent_map + 30 and online_map) or percent_map == 100:
    # 	print('...generating map')
    # 	generate_hilbert_map(
    # 	  model,
    # 	  map_resolution,
    # 	  [xlim[0], xlim[1], ylim[0], ylim[1]],
    # 	  "hilbert_map" + str(percent_map) + " .png", 
    #      percent_map 
    # 	)
    # 	old_percent_map = percent_map
    count += 1 
	 

def main():
  # Set command line argument parsing 
  parser = argparse.ArgumentParser(description='Ty Demo Hilbert Map')
  parser.add_argument("--data_dir", help="Logfile in CARMEN format to process",\
    #default="../datasets/intel.gfs.log")
    default="../datasets/fr-campus-20040714.carmen.gfs.log")
  parser.add_argument("--log_dir", help="Logfile to save",\
    #default="../datasets/intel.gfs.log")
    default="../logs/")

  parser.add_argument("--feature", choices=["sparse", "fourier", "nystroem", "GP"],\
    help="The feature to use", \
    default="GP")
  parser.add_argument("--components", \
    default=1000,\
    type=int,\
    help="Number of components used with the feature")
  parser.add_argument("--distance_cutoff",\
    default=0.001,\
    type=float,\
    help="Value below which a kernel value will be set to 0")
  parser.add_argument("--resolution",\
    default=0.1,\
    type=float,\
    help="Grid cell resolution of the map")

  # Parameters of GP 
  sigma = 0.1 
  v0 = 1 


  args = parser.parse_args()

  # Load the data and split it into training and testing data 
  if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

  train_data, test_data = util.create_test_train_split(args.data_dir, 0.1)	
  # Create a simple grid map 
  #simple_grid(train_data, 'fr-campus_simple_grid.png')
  
  # Train the desired model on the data 
  if args.feature == 'GP':
    model = train_GP(train_data, args.components, sigma, v0, args.distance_cutoff, args.resolution, max_iter=1)
  # else:
  # 	model = train_incremental_hm(train_data, args.components, args.gamma, args.feature, visual_map=False)

  # Produce grid map based on the trained model
  xlim, ylim = util.bounding_box(train_data["poses"], 10.0)
  generate_GP_map(
          model,
          args.resolution,
          [xlim[0], xlim[1], ylim[0], ylim[1]],
          os.path.join(args.log_dir, "GP_output_map.png")
  )

if __name__=="__main__":
	main()
