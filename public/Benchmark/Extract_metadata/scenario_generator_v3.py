import sys
from contextlib import closing
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from datetime import datetime
from time import time
import traceback
import os
import errno
from scipy.spatial import distance
from scipy.cluster import hierarchy as hier
from scipy.spatial import distance as ssd
from scipy.cluster.hierarchy import dendrogram, linkage
import sklearn.metrics.pairwise
import random
from sklearn.model_selection import KFold
from collections import OrderedDict
import operator

def getKey(item):
    return item[1]
vertexmap = OrderedDict()

def getDegree(x):
	command = "degree" + " " + str(x)
	degree = int(os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read())
	return degree

def get_node_id(value):
	command = "node_id" + " " + value
	return os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(
        description=__doc__, 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
	parser.add_argument('basedir', metavar='base', help='experiment base directory')
	parser.add_argument('-r', type=str, required=True,
            dest='relation', help='Relation')
	parser.add_argument('-sc', type=int, required=True,
            dest='size_scenario', help='Size of Scenario')
	parser.add_argument('-c', type=str, required=True,
        dest='cluster', help='cluster')
	args = parser.parse_args()





	with closing(open(args.basedir + "graphs/edge_dict.tsv")) as f:
		nodes = f.readlines()
		edge_types = [nodes[i].rstrip('\n').split('\t') for i in range(len(nodes))]
		vertexmap = OrderedDict(( (int(i), edge) for i, edge in sorted(edge_types, key=getKey) ))
		del edge_types, nodes

	output_filename = args.basedir + "splits/" +args.relation +  "/" + "cluster_" + args.cluster + "_scenario.tsv"

	if  os.path.isfile(output_filename):
		print "File exist !"
		sys.exit(0)

	with closing(open(args.basedir + "splits/" + args.relation + "/" + args.relation + ".tsv")) as f:
		nodes = f.readlines()
		raw_entities = [nodes[i].rstrip('\n').split('\t')[:3] for i in range(len(nodes))]
		# if (len(raw_entities) > 10000):
		# entities = raw_entities[:8000]
		# elif (len(raw_entities) > 5000):
		# 	entities = raw_entities[:5000]
		# else:
		# 	entities = raw_entities[:1500]
		entities = raw_entities


	with closing(open(args.basedir + "splits/" + args.relation + "/" + args.relation + "_id.tsv")) as f:
		nodes = f.readlines()
		raw_entities_id = [map(int, nodes[i].rstrip('\n').split('\t')[:2]) for i in range(len(nodes))]
		# if (len(raw_entities_id)>10000):
		# entities_id = raw_entities_id[:8000]
		# elif (len(raw_entities_id)>5000):
		# 	entities_id = raw_entities_id[:5000]
		# else:
		# 	entities_id = raw_entities_id[:1500]
		entities_id = raw_entities_id
		unique_id = sorted(set(sum(entities_id, [])))

	if not os.path.isfile(args.basedir + "splits/" + args.relation + "/" + args.relation + "_embedding.npy"):
		if 	os.path.isfile(args.basedir + "graphs/" +  "Full_TransE_entity_vec.txt" ):
			file_names = open(args.basedir + "graphs/" + "Full_TransE_entity_vec.txt" , 'r')
			file_name = file_names.readline()
			count = 1
			index = 0

			flag = True
			map_embed = {}
			while (file_name != "") and flag:
				file_name = file_names.readline()
				if count == unique_id[index]:
					em = map(float, file_name.rstrip('\n').split(' ')[:100])
					map_embed[count] = em
					index = index + 1
					if (index == len(unique_id)):
						flag = False
				count = count + 1
			file_names.close()
			entities_embed = [sum([map_embed[i], map_embed[j]], []) for i, j in entities_id]
			entities_embed = np.array(entities_embed)
			np.save(args.basedir + "splits/" + args.relation + "/" + args.relation + "_embedding.npy", entities_embed)
		else:
			print "Embedding of this relation not found !!!"
			sys.exit(0)

	else:
		entities_embed = np.load(args.basedir + "splits/" + args.relation + "/" + args.relation + "_embedding.npy")

	
	if not os.path.isfile(args.basedir + "splits/" + args.relation + "/" + args.relation + "_clustering.npy"):
		score = sklearn.metrics.pairwise.pairwise_distances(entities_embed[:,:100], entities_embed[:,:100], n_jobs=1) +  \
				sklearn.metrics.pairwise.pairwise_distances(entities_embed[:,100:], entities_embed[:,100:], n_jobs=1) + \
				sklearn.metrics.pairwise.pairwise_distances(entities_embed[:,100:], entities_embed[:,:100], n_jobs=1) + \
				sklearn.metrics.pairwise.pairwise_distances(entities_embed[:,:100], entities_embed[:,100:], n_jobs=1) 
		score = score/4
		upper_i = np.triu_indices(len(entities_embed),1)
		score = score[upper_i]
		np.save(args.basedir + "splits/" + args.relation + "/" + args.relation + "_clustering.npy", score)
	else:
		score = np.load(args.basedir + "splits/" + args.relation + "/" + args.relation + "_clustering.npy")

	Z = linkage(score,  method="average")


	heterogenity = []
	l_cut = np.arange(np.ceil(Z[0,2]), np.ceil(Z[-1,2]), 0.1)
	for i_cut in l_cut:
		cut = hier.fcluster(Z, i_cut, criterion="distance")
		unique_cut = np.unique(cut)
		heterogenity.append(len(unique_cut))
	
	acceleration = -np.diff(heterogenity)
	acceleration = acceleration*l_cut[1:]

	threshold = l_cut[acceleration.argmax() + 1]

	cut = hier.fcluster(Z, threshold, criterion="distance")
	
	(values,counts) = np.unique(cut,return_counts=True)
	sort_index = np.argsort(-counts)
	values = values[sort_index]
	counts = counts[sort_index]

	error = ""
	flag_possibility = True
	i_cluster = (int) (args.cluster)

	random.seed(0)
	positive_set = []

	## STARTING CREATING SCENARIO
	if (np.array(entities)[cut==values[i_cluster]].shape[0] > args.size_scenario):
	# if True:
		# print "Size of scenario: " + str(np.array(entities)[cut==values[i_cluster]].shape[0])
		cluster_tmp  = np.array(entities)[cut==values[i_cluster]]

		# cluster_id_tmp = np.array(entities_id)[cut==values[i_cluster]]
		
		# index = np.random.choice(len(entities), 900, replace=False)  
		# cluster_tmp  = np.array(entities)[index]
		cluster_tmp = cluster_tmp.tolist()
		for positive_sample in cluster_tmp:
			# if (positive_sample in overlapping_pair ):
			if (positive_sample[1] != positive_sample[0])  and ([positive_sample[1], positive_sample[0]] not in [x[:2] for x in positive_set]):
				positive_sample.append(i_cluster)
				positive_set.append(positive_sample)

		random.seed(100)

		random.shuffle(positive_set)
		print "Size of positive set: " + str(len(positive_set))

		negative_set = []
		count = 0
		print "... Major negative example..."

		for positive_sample in positive_set:
			print "Positive sample: " + str(count+1)

			sub_id = get_node_id(positive_sample[0])
			obj_id = get_node_id(positive_sample[1])
			
			command = "ontology" + " " + obj_id
			o_onto = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()
			o_onto = set(o_onto.rstrip("\n").rstrip(",").split(","))

			command = "ontology" + " " + sub_id
			s_onto = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()
			s_onto = set(s_onto.rstrip("\n").rstrip(",").split(","))

			command = "neighbor" + " " + sub_id + " " + obj_id + " " + positive_sample[2] + " " + "TRUE"
			neighbor = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()	

			neighbor = neighbor.split('\n')[1:-1]
			neighbor = [pair.split('\t') for pair in neighbor]
			# t1 = time()
			neighbor_s = []
			for node in neighbor:
				# command = "ontology" + " " + node[0]
				# node_onto = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()
				# node_onto = set(node_onto.rstrip("\n").rstrip(",").split(","))
				# if (len(o_onto.intersection(node_onto)) >= min(4, len(o_onto))):
				# 	neighbor_s.append(node[1])	
				# if (len(neighbor_s)>2):
				# 	break
				neighbor_s.append(node[1])

			# print " -- 1: " + str(len(neighbor_s))

			if (len(neighbor_s) > 0 ):
				# neighbor_s = neighbor_s[:2]
				for entity in neighbor_s:

					if ([positive_sample[0], entity, positive_sample[2]] not in raw_entities):
						t1 = time()
						command = "hpath" + " " + get_node_id(positive_sample[0]) + " " + get_node_id(entity) + " "  +  "10000" + " " + "3" + " " + "F" + " " + "P"
						features = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()
						features =  features.split("\n")[1:-1] 
						# print " -- 2: " + str(time()-t1)
						
						random.shuffle(features)
						# random_prob = []
						stop = False
						t1 = time()
						for path in features:
							
							path = path.rstrip(',').split(',')
							s = [ ('(-1)' if (int(x) < 0) else '') + vertexmap[abs(int(x))] for x in path]
							path_name = ','.join(s)

							node = positive_sample[0]
							for i in range(len(path)):
								r = path[i]
								if (int(r) > 0):
									command = "neighborwithrel" + " " + node + " " + vertexmap[int(r)] + " " + "TRUE"
								else:
									command = "neighborwithrel" + " " + node + " " + vertexmap[-int(r)] + " " + "FALSE"
								relnbrs = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()			
								relnbrs =  relnbrs.split('\n')[1:-1]
								relnbrs = [pair.split('\t')[1] for pair in relnbrs]

								n_nbrs = len(relnbrs)
								if n_nbrs == 0:
									# node = "NULL"
									break # restart random walk
								else:
									random.seed(100)
									random.shuffle(relnbrs)
									if (i == (len(path) -1 )):
										# node = "NULL"
										for tmp in relnbrs[:1000]:
											command = "ontology" + " " + get_node_id(tmp)
											node_onto = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()
											node_onto = set(node_onto.rstrip("\n").rstrip(",").split(","))
											if (len(o_onto.intersection(node_onto)) >= min(4, len(o_onto))):
												if ([tmp, positive_sample[0]] not in [x[:2] for x in negative_set]) and ([positive_sample[0], tmp] not in [x[:2] for x in negative_set]):
													if  ([positive_sample[0], tmp, positive_sample[2]] not in raw_entities) and ([tmp, positive_sample[0], positive_sample[2]] not in raw_entities) and \
															(tmp!= positive_sample[0]) and \
																([tmp, positive_sample[0]] not in [x[:2] for x in negative_set]) and \
																	([positive_sample[0],  tmp] not in [x[:2] for x in negative_set]) :
														# node = tmp
														negative_example = [positive_sample[0], tmp, positive_sample[2], path_name, positive_sample[3]]
														negative_set.append(negative_example)
										
														stop = True
														break

									else:
										# np.random.seed(100)
										node = np.random.choice(relnbrs, 1)[0] # pick 1 nbr uniformly at random
							# if stop:
							# 	break
						

			command = "neighbor" + " " + obj_id + " " + sub_id + " " + positive_sample[2] + " " + "FALSE"
			neighbor = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()			
			neighbor =  neighbor.split('\n')[1:-1]	
			neighbor = [pair.split('\t') for pair in neighbor]

			neighbor_o = []
			for node in neighbor:
				# command = "ontology" + " " + node[0]
				# node_onto = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()
				# node_onto = set(node_onto.rstrip("\n").rstrip(",").split(","))
				# if (len(s_onto.intersection(node_onto)) >= min(4, len(s_onto))):
				# 	neighbor_o.append(node[1])	
				# if (len(neighbor_o)>2):
				# 	break
				neighbor_o.append(node[1])
			# print " -- 2: " + str(len(neighbor_o))
			if (len(neighbor_o) > 0):
				# neighbor_o = neighbor_o[:2]
				for entity in neighbor_o:
					if ([entity, positive_sample[1], positive_sample[2]] not in raw_entities):
						command = "hpath" + " " + get_node_id(entity) + " " + get_node_id(positive_sample[1]) +  " "  +  "10000" + " " + "3" + " " + "F" + " " + "P"
						features = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()
						features =  features.split("\n")[1:-1] 

						# random_prob = []
						random.shuffle(features)
						stop = False
						for path in features:
							path = path.rstrip(',').split(',')
							s = [ ('(-1)' if (int(x) < 0) else '') + vertexmap[abs(int(x))] for x in path]
							path_name = ','.join(s)

							node = positive_sample[1]
							for i in reversed(range(len(path))):
								r = path[i]
								if (int(r) > 0):
									command = "neighborwithrel" + " " + node + " " + vertexmap[int(r)] + " " + "FALSE"
								else:
									command = "neighborwithrel" + " " + node + " " + vertexmap[-int(r)] + " " + "TRUE"
								relnbrs = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()			
								relnbrs =  relnbrs.split('\n')[1:-1]
								relnbrs = [pair.split('\t')[1] for pair in relnbrs]
								n_nbrs = len(relnbrs)
								if n_nbrs == 0:
									node = "NULL"
									break # restart random walk
								else:
									random.seed(100)
									random.shuffle(relnbrs)
									if (i == 0):
										node = "NULL"
										for tmp in relnbrs[:1000]:
											command = "ontology" + " " + get_node_id(tmp)
											node_onto = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()
											node_onto = set(node_onto.rstrip("\n").rstrip(",").split(","))
											if (len(s_onto.intersection(node_onto)) >= min(4, len(s_onto))):
												if ([tmp, positive_sample[1]] not in [x[:2] for x in negative_set]) and ([positive_sample[1], tmp] not in [x[:2] for x in negative_set]):
													if  ([tmp, positive_sample[1], positive_sample[2]] not in raw_entities) and ([positive_sample[1], tmp, positive_sample[2]] not in raw_entities) and \
															(tmp!= positive_sample[1])  and \
																([tmp, positive_sample[0]] not in [x[:2] for x in negative_set]) and \
																	([positive_sample[0],  tmp] not in [x[:2] for x in negative_set]) :
														negative_example = [tmp, positive_sample[1], positive_sample[2], path_name, positive_sample[3]]
														negative_set.append(negative_example)
													
														stop = True
														# node = tmp
														break

									else:
										# np.random.seed(100)
										node = np.random.choice(relnbrs, 1)[0] # pick 1 nbr uniformly at random

							# if stop:
							# 	break	
						

			count = count + 1

		random.seed(100)
		random.shuffle(negative_set)
		negative_set = negative_set[:len(positive_set)]

	else:
		flag_possibility = False
		error = "Cannot create scenario: Not enough suitable positive examples..."
		sys.exit(0)

	
	if not os.path.isfile(output_filename):
		if not os.path.exists(os.path.dirname(output_filename)):
			try:
				os.makedirs(os.path.dirname(output_filename))
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise

	ids =  open(output_filename, 'w')
	if (flag_possibility):
		ids.write("Subject" + "\t" + "Object" + "\t" + "Relation" + "\t" + "Label" + "\t" + "Origin_Relation" + "\t" + "Cluster" + "\n")
		for node in positive_set:
			ids.write(node[0] + "\t" + node[1] + "\t" + node[2] + "\t" + "1" + "\t" + "No" + "\t" + str(node[3]) + "\n")
		for node in negative_set:
			ids.write(node[0] + "\t" + node[1] + "\t" + node[2] + "\t" + "-1" + "\t" + node[3] + "\t"+ str(node[4]) +"\n")
		ids.close()
		print "Successfully generated !!!"
	else:
		print error
