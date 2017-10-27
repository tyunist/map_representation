import math 
import numpy as np 
import random 
import time 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score, roc_curve 

import hilbert_map as hm 

def parse_carmen_log(fname):
	"""Parses a CARMEN log file and extracts poses and laser scans.

	:param fname the path to the log file to parse
	:return poses and scans extracted from the log file
	"""
	poses = [] 
	scans = [] 
	for line in open(fname):
		if line.startswith('FLASER'):
			arr = line.split()
			count = int(arr[1])
			poses.append([float(v) for v in arr[-9:-6]])
			scans.append([float(v) for v in arr[2:2+count]])
	return poses, scans 

def create_test_train_split(logfile, percentage=0.1, sequence_length=40):
	"""Creates a testing and training dataset from the given logfile.

	:param logfile the file to parse
	:param percentage the percentage to use for testing
	:param sequence_length the number of subsequent scans to remove for
	    the testing data
	:return training and testing datasets containing the posts and scans
	"""
	# parse the log file 
	poses, scans = parse_carmen_log(logfile)

	# create training and testing splits 
	groups = int((len(poses)*percentage) / sequence_length)
	test_indices = []
	group_count = 0

	while group_count < groups:
		start = random.randint(0, len(poses) - sequence_length)
		if start in test_indices or (start + sequence_length) in test_indices:
			continue 
		test_indices.extend(range(start, start + sequence_length))
		group_count += 1

	training = {"poses": [], "scans": []}
	testing = {"poses": [], "scans": []}
	  
	for i in range(len(poses)):
	    if i in test_indices:
	        testing["poses"].append(poses[i])
	        testing["scans"].append(scans[i])
	    else:
	        training["poses"].append(poses[i])
	        training["scans"].append(scans[i])

	return training, testing

def bounding_box(data, padding=5.0):
  """Returns the bounding box to the given 2d data.

  :param data the data for which to find the bounding box
  :param padding the amount of padding to add to the extreme values
  :return x and y limits
  """
  dimensions = len(data[0])
  limits = []
  for dim in range(dimensions):
      limits.append((
          np.min([entry[dim] for entry in data]) - padding,
          np.max([entry[dim] for entry in data]) + padding
      ))
  assert(len(limits) > 1)
  return limits[0], limits[1]


def sampling_coordinates(x_limits, y_limits, count):
  """Returns an array of 2d grid sampling locations.

  :params x_limits x coordinate limits
  :params y_limits y coordinate limits
  :params count number of samples along each axis
  :return list of sampling coordinates
  """	
  coords = [] 
  for i in np.linspace(x_limits[0], x_limits[1], count):
  	for j in np.linspace(y_limits[0], y_limits[1], count):
  		coords.append([i, j])
  return np.array(coords)
 

def free_space_points(distance, pose, angle):
	"""Samples points randomly along a scan ray.

	:param distance length of the ray
	:param pose the origin of the ray
	:param angle the angle of the ray from the position
	:return list of coordinates in free space based on the data
	"""
	points = [] 
	count = max(1, int(distance/2))
	for _ in range(count):
		r = random.uniform(0.0, max(0.0, distance - 0.1))
		points.append([
			pose[0] + r*math.cos(angle), 
			pose[1] + r*math.sin(angle)])
	return points 



def normalize_angle(angle):
	"""Normalize the angle to the range [-PI, PI]
	:params angle the angle to normalize 
	:return  normalized angle"""
	center = 0.0 
	n_angle = angle - 2*math.pi*math.floor((angle + math.pi - center)/(2*math.pi))
	assert(-math.pi <= n_angle <= math.pi)
	return n_angle 	

def data_generator(poses, scans, step=1):
	"""Generator which returns data for each scan
	:params poses the sequence of poses
	:params scans the sequence of scans observed at each pose
	:params step the step size to use in the iteration
	:return 2d coordinates and labels for the data generated for individual pose 
	and scan pairs
	# pose = (x, y, theta)
	"""

	# number of scans: len(scans[0])
	angle_increment = math.pi/len(scans[0])
	for i in range(0, len(poses), step):
		pose = poses[i]
		ranges = scans[i]
		points = [] 
		labels = [] 

		for range_numb, dist in enumerate(ranges):
			# Ignore max range readings 
			if dist > 40:
				continue 
			angle = normalize_angle(pose[2] - math.pi + i*angle_increment + math.pi/2.0)
  
  		# Add laser endpoint 
			points.append([
				pose[0] + dist*math.cos(angle), 
				pose[1] + dist*math.sin(angle)])
			labels.append(1)

			# Add in between points 
			free_points = free_space_points(dist, pose, angle)
			points.extend(free_points)
			for coord in free_points:
				labels.append(0)
		yield np.array(points), np.array(labels)


def scatter2D(points):
	plt.figure()
	plt.scatter(points[:,0], points[:,1])