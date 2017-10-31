import argparse 
import copy 
import math 
import matplotlib.pyplot as plt 
import numpy as np 

import sys, pdb
import hilbert_map as hm 

import util 

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
 
def generate_hilbert_map(model, resolution, limits, fname, verbose=True, percent_map=100):
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
    batch_size = 100
    old_intercept = copy.deepcopy(model.classifier.intercept_)
    model.classifier.intercept_ = 0.1 * model.classifier.intercept_
    while offset < len(sample_coords):
        # if isinstance(model, hm.IncrementalHilbertMap):
        #     query = model.sampler.transform(sample_coords[offset:offset+batch_size])
        #     predictions.extend(model.classifier.predict_proba(query)[:, 1])
        if isinstance(model, hm.SparseHilbertMap):
            predictions.extend(model.classify(sample_coords[offset:offset+batch_size])[:, 1])

        if verbose:
            sys.stdout.write("\rQuerying model: {: 6.2f}%".format(offset / float(len(sample_coords)) * 100))
            sys.stdout.flush()
        offset += batch_size
    if verbose:
        print("")
    predictions = np.array(predictions)
    model.classifier.intercept_ = old_intercept

    # Turn predictions into a matrix for visualization
    mat = predictions.reshape(x_count, y_count)
    plt.clf()
    plt.title("Occupancy map %d %%"%percent_map)
    plt.imshow(mat.transpose()[::-1, :])
    plt.colorbar()
    plt.savefig(fname)
    plt.show()
    plt.pause(0.05)


def train_sparse_hm(data, components, gamma, distance_cutoff, map_resolution=None, online_map=False):
	"""Trains a hilbert map model using the sparse feature.

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

	model = hm.SparseHilbertMap(centers, gamma, distance_cutoff)

	# train the model with the data 
	count = 0 
	old_percent_map = 0 
	for data, label in util.data_generator(poses, scans):
		print '...'
		# print 'data', data, 'label', label 
		percent_map = count/float(len(poses)) * 100
		model.add(data, label)
		sys.stdout.write("\rTraining model:{: 6.2f}%".format(percent_map))
		sys.stdout.flush()
		sys.stdout.write("\rold_percent_map:{: 6.2f}%".format(old_percent_map))
		sys.stdout.flush()
		print('\n...current %%: ', percent_map)
		if (percent_map > old_percent_map + 30 and online_map) or percent_map == 100:
			print('...generating map')
			generate_hilbert_map(
			  model,
			  map_resolution,
			  [xlim[0], xlim[1], ylim[0], ylim[1]],
			  "hilbert_map" + str(percent_map) + " .png", 
        percent_map 
			)
			old_percent_map = percent_map
		count += 1 
	 

def main():
  # Set command line argument parsing 
  parser = argparse.ArgumentParser(description='Ty Demo Hilbert Map')
  parser.add_argument("--log_dir", help="Logfile in CARMEN format to process",\
    #default="../datasets/intel.gfs.log")
    default="map_representation/datasets/fr-campus-20040714.carmen.gfs.log")

  parser.add_argument("--feature", choices=["sparse", "fourier", "nystroem"],\
    help="The feature to use", \
    default="sparse")
  parser.add_argument("--components", \
    default=1000,\
    type=int,\
    help="Number of components used with the feature")
  parser.add_argument("--gamma", \
    default=1,\
    type=float,\
    help="Gamma value used in the RBF kernel")
  parser.add_argument("--distance_cutoff",\
    default=0.001,\
    type=float,\
    help="Value below which a kernel value will be set to 0")
  parser.add_argument("--resolution",\
    default=0.1,\
    type=float,\
    help="Grid cell resolution of the map")

  args = parser.parse_args()

  # Load the data and split it into training and testing data 

  train_data, test_data = util.create_test_train_split(args.log_dir, 0.1)	
  # Create a simple grid map 
  #simple_grid(train_data, 'fr-campus_simple_grid.png')
  
  # Train the desired model on the data 
  if args.feature == 'sparse':
    model = train_sparse_hm(train_data, args.components, args.gamma, args.distance_cutoff, args.resolution)
  # else:
  # 	model = train_incremental_hm(train_data, args.components, args.gamma, args.feature, visual_map=False)

  # Produce grid map based on the trained model
  xlim, ylim = util.bounding_box(train_data["poses"], 10.0)
  generate_hilbert_map(
          model,
          args.resolution,
          [xlim[0], xlim[1], ylim[0], ylim[1]],
          "hilbert_map.png"
  )

if __name__=="__main__":
	main()
