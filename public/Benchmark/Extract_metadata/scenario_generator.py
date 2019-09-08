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
	parser.add_argument('-t', type=str, required=True,
            dest='type', help='Train or test scenario')
	parser.add_argument('-hm', type=float, required=True,
            dest='homo', help='Homogeneity')
	parser.add_argument('-cs', type=float, required=True,
            dest='consis', help='Consistency')
	parser.add_argument('-sc', type=int, required=True,
            dest='size_scenario', help='Size of Scenario')
	parser.add_argument('-pn', type=float, required=True,
            dest='pos_neg', help='Ratio of positive/negative')
	parser.add_argument('-p', type=str, required=True,
            dest='popular', help='Popularity')
	parser.add_argument('-f', type=str, required=True,
            dest='train_file_ifexist', help='Train file')
	args = parser.parse_args()

	if (args.type == "train"):
		train_scenario_tocheck = []
	elif (args.type == "test"):
		if os.path.isfile(args.train_file_ifexist):
			with closing(open(args.train_file_ifexist)) as f:
				nodes = f.readlines()
				train_scenario_tocheck = [nodes[i].rstrip('\n').split('\t')[:2] for i in range(len(nodes))]
					
		else:
			print("Should run train mode first to create train scenario !!!")
			sys.exit(0)
	else:
		print("Cannot determine mode of operation: train/test ?")
		sys.exit(0)

	output_filename = args.basedir + "splits/" +args.relation +  "/" + args.type + "_c" + str(args.consis) + "_h" + str(args.homo) + "_s" + str(args.size_scenario) + "_r" + str(args.pos_neg) + "_" + args.popular + "_scenario.tsv"
	if os.path.isfile(output_filename):
		print("Successfully generated !!!")
		sys.exit(0);
				
	with closing(open(args.basedir + "graphs/edge_dict.tsv")) as f:
		nodes = f.readlines()
		edge_types = [nodes[i].rstrip('\n').split('\t') for i in range(len(nodes))]
		vertexmap = OrderedDict(( (int(i), edge) for i, edge in sorted(edge_types, key=getKey) ))
		del edge_types, nodes

	if not os.path.isfile(args.basedir + "splits/" + args.relation + "/" + args.relation + ".tsv"):
		print("Not enough triples for this predicate. Change to another !")
		sys.exit(0)

	with closing(open(args.basedir + "splits/" + args.relation + "/" + args.relation + ".tsv")) as f:
		nodes = f.readlines()
		raw_entities = [nodes[i].rstrip('\n').split('\t')[:3] for i in range(len(nodes))]
		# if (len(raw_entities) > 10000):
		entities = raw_entities[:8000]

		# elif (len(raw_entities) > 5000):
			# entities = raw_entities[:5000]
		# else:
			# entities = raw_entities[:1500]
		# entities = raw_entities


	with closing(open(args.basedir + "splits/" + args.relation + "/" + args.relation + "_id.tsv")) as f:
		nodes = f.readlines()
		raw_entities_id = [list(map(int, nodes[i].rstrip('\n').split('\t')[:2])) for i in range(len(nodes))]
		popularity_G = [list(map(int, nodes[i].rstrip('\n').split('\t')[3:])) for i in range(len(nodes))]
				# if (len(raw_entities_id)>10000):
		entities_id = raw_entities_id[:8000]
		popularity_G = np.array(popularity_G[:8000])
		# elif (len(raw_entities_id)>5000):
			# entities_id = raw_entities_id[:5000]
		# else:
			# entities_id = raw_entities_id[:1500]
		# entities_id = raw_entities_id
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
					em = list(map(float, file_name.rstrip('\n').split(' ')[:100]))
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
			print("Embedding of this relation not found !!!")
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
	
	homogeneity_maj = 0
	consistency_maj = []

	error = ""
	flag_possibility = True

	random.seed(0)

	## STARTING CREATING SCENARIO
	num_positive = int(np.ceil(args.size_scenario*args.pos_neg/(1+args.pos_neg)))

	maj_len = (int) (args.homo*num_positive)
	mij_len = num_positive - maj_len
	mij_mask = np.ones(len(cut), dtype = bool)

	# consistency_maj.append(len(args.similar_relation))
	consistency_maj.append("*")
	i_cluster = 0
	positive_set = []

	# print "... Major positive example..."
	while (len(positive_set) < maj_len) and (i_cluster<len(values)):
		cluster_tmp  = np.array(entities)[cut==values[i_cluster]]

		cluster_id_tmp = np.array(entities_id)[cut==values[i_cluster]]

		if (args.popular == "popular"):
			cluster_G = popularity_G[cut==values[i_cluster]]
			# cluster_G = np.min(cluster_G, axis=1)*(1+np.max(cluster_G,axis=1)/np.mean(cluster_G))
			cluster_G = np.min(cluster_G, axis=1)
			indices = np.argsort(-cluster_G)
			cluster_tmp = cluster_tmp[indices]
			cluster_tmp = cluster_tmp.tolist()
		elif (args.popular == "nopopular"):
			cluster_G = popularity_G[cut==values[i_cluster]]
			# cluster_G = np.min(cluster_G, axis=1)*(1+np.max(cluster_G,axis=1)/np.mean(cluster_G))
			cluster_G = np.min(cluster_G, axis=1)
			indices = np.argsort(cluster_G)
			cluster_tmp = cluster_tmp[indices]	
			cluster_tmp = cluster_tmp.tolist()
		elif (args.popular == "random"):
			random.seed(100)
			cluster_tmp = cluster_tmp.tolist()
			random.shuffle(cluster_tmp)
		mij_mask[cut==values[i_cluster]] = False
		for positive_sample in cluster_tmp:
			if (positive_sample[1] != positive_sample[0]) and (positive_sample[:2] not in train_scenario_tocheck) and ([positive_sample[1], positive_sample[0]] not in train_scenario_tocheck) and ([positive_sample[1], positive_sample[0]] not in [x[:2] for x in positive_set]):
				positive_sample.append(i_cluster)
				positive_set.append(positive_sample)
				if (len(positive_set) >= maj_len):
					break
		i_cluster = i_cluster + 1 

	positive_set = positive_set[:maj_len]
	homogeneity_maj = i_cluster

	mij_cut = np.array(entities)[mij_mask]
	mij_id_cut = np.array(entities_id)[mij_mask]
	mij_label = cut[mij_mask]
	cluster_G = popularity_G[mij_mask]

	# print "... Minor positive example..."
	if (args.popular == "popular"):
		# cluster_G = np.min(cluster_G, axis=1)*(1+np.max(cluster_G,axis=1)/np.mean(cluster_G))
		cluster_G = np.min(cluster_G, axis=1)
		indices = np.argsort(-cluster_G)
		mij_cut = mij_cut[indices]
		mij_cut = mij_cut.tolist()
		mij_label = mij_label[indices].tolist()
	elif (args.popular == "nopopular"):
		# cluster_G = np.min(cluster_G, axis=1)*(1+np.max(cluster_G,axis=1)/np.mean(cluster_G))
		cluster_G = np.min(cluster_G, axis=1)
		indices = np.argsort(cluster_G)
		mij_cut = mij_cut[indices]	
		mij_cut = mij_cut.tolist()
		mij_label = mij_label[indices].tolist()
	elif (args.popular == "random"):
		random.seed(100)
		mij_cut = mij_cut.tolist()
		random.shuffle(mij_cut)
		random.seed(100)
		mij_label = mij_label.tolist()
		random.shuffle(mij_label)
	
	unique_label, label_counts = np.unique(mij_label, return_counts = True)
	label_prior = np.argsort(-label_counts)
	unique_label = unique_label[label_prior]
	label_counts = label_counts[label_prior]

	if (mij_len > len(mij_cut)):
		mij_len = len(mij_cut)

	index_of_label = {}

	for label_ in unique_label:
		index_of_label[label_] = np.where(np.array(mij_label) == label_)[0]

	rearranged_index = []
	i = 0
	while (i<label_counts[0]):
		for label_ in unique_label:
			if (i<len(index_of_label[label_])):
				rearranged_index.append([index_of_label[label_][i], np.where(unique_label==label_)[0][0]+homogeneity_maj])
		i = i + 1
	i = 0
	while ((len(positive_set) < num_positive) and (i<len(mij_cut))):
		non_homo_sample = mij_cut[rearranged_index[i][0]]
		if (non_homo_sample[1] != non_homo_sample[0]) and (non_homo_sample[:2] not in train_scenario_tocheck) and ([non_homo_sample[1], non_homo_sample[0]] not in train_scenario_tocheck) and ([non_homo_sample[1], non_homo_sample[0]] not in [x[:2] for x in positive_set]):
			non_homo_sample.append(rearranged_index[i][1])
			positive_set.append(non_homo_sample)
		i = i + 1	

	if (len(positive_set) < num_positive):
		flag_possibility = False
		error = "Cannot create scenario: Not enough suitable positive examples..."
	else:
		random.seed(100)

		random.shuffle(positive_set)

		negative_set = []
		negative_len = args.size_scenario - num_positive
		maj_len = (int) ((1.0-args.consis)*negative_len)
		mij_len = negative_len - maj_len
		# print "... Major negative example..."
		if (maj_len > 0):
			negative_cluster = []
			for positive_sample in positive_set:

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
					neighbor_s.append(node[1])

				# print " -- 1: " + str(time()-t1)

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
																(tmp!= positive_sample[0]) and ([positive_sample[0], tmp] not in train_scenario_tocheck) and \
																	([tmp, positive_sample[0]] not in train_scenario_tocheck) and ([tmp, positive_sample[0]] not in [x[:2] for x in negative_set]) and \
																		([positive_sample[0],  tmp] not in [x[:2] for x in negative_set]) :
															# node = tmp
															negative_example = [positive_sample[0], tmp, positive_sample[2], path_name, positive_sample[3]]
															negative_set.append(negative_example)
															negative_cluster.append(positive_sample[3])
															stop = True
															break

										else:
											# np.random.seed(100)
											node = np.random.choice(relnbrs, 1)[0] # pick 1 nbr uniformly at random
								if stop:
									break

				command = "neighbor" + " " + obj_id + " " + sub_id + " " + positive_sample[2] + " " + "FALSE"
				neighbor = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()			
				neighbor =  neighbor.split('\n')[1:-1]	
				neighbor = [pair.split('\t') for pair in neighbor]

				neighbor_o = []
				for node in neighbor:
					neighbor_o.append(node[1])
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
																(tmp!= positive_sample[1]) and ([positive_sample[1], tmp] not in train_scenario_tocheck) and \
																	([tmp, positive_sample[1]] not in train_scenario_tocheck)  :
															negative_example = [tmp, positive_sample[1], positive_sample[2], path_name, positive_sample[3]]
															negative_set.append(negative_example)
															negative_cluster.append(positive_sample[3])
															stop = True
															# node = tmp
															break

										else:
											# np.random.seed(100)
											node = np.random.choice(relnbrs, 1)[0] # pick 1 nbr uniformly at random

								if stop:
									break	
	
			random.seed(100)
			random.shuffle(negative_set)
			negative_set = negative_set[:maj_len]

			positive_set_no_ = [x[:2] for x  in positive_set]
			positive_set_no_label = set(sum(positive_set_no_, []))
			repeat = 0
			i_cluster = 0
			# print "... Minor negative example..."
			while ((len(negative_set) < maj_len)):
				extra_positive_set = np.array(entities).tolist()
				# extra_positive_set = [sample for sample in extra_positive_set if ((sample[0] not in positive_set_no_label) and  (sample[1] not in positive_set_no_label))]
				extra_positive_set = [sample for sample in extra_positive_set if (sample not in positive_set_no_)]

				random.seed(100)
				random.shuffle(extra_positive_set)
				for positive_sample in extra_positive_set:
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

					neighbor_s = []
					for node in neighbor:
						neighbor_s.append(node[1])

					if (len(neighbor_s) > 0 ):
						# neighbor_s = neighbor_s[:2]
						for entity in neighbor_s:
							if ([positive_sample[0], entity, positive_sample[2]] not in raw_entities):
								command = "hpath" + " " + get_node_id(positive_sample[0]) + " " + get_node_id(entity) + " "  +  "10000" + " " + "3" + " " + "F" + " " + "P"
								features = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()
								features =  features.split("\n")[1:-1] 

								# random_prob = []
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
															if ([positive_sample[0], tmp, positive_sample[2]] not in raw_entities) and ([tmp, positive_sample[0], positive_sample[2]] not in raw_entities) and \
																	(tmp!= positive_sample[0]) and ([positive_sample[0], tmp] not in train_scenario_tocheck) and \
																		([tmp, positive_sample[0]] not in train_scenario_tocheck)  :
																# node = tmp
																negative_example = [positive_sample[0], tmp, positive_sample[2], path_name, "#"]
																negative_set.append(negative_example)
																negative_cluster.append("#")
																break

											else:
												# np.random.seed(100)
												node = np.random.choice(relnbrs, 1)[0] # pick 1 nbr uniformly at random

					if (len(negative_set) >= maj_len):
						break

					command = "neighbor" + " " + obj_id + " " + sub_id + " " + positive_sample[2] + " " + "FALSE"
					neighbor = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()			
					neighbor =  neighbor.split('\n')[1:-1]	
					neighbor = [pair.split('\t') for pair in neighbor]

					neighbor_o = []
					for node in neighbor:
						neighbor_o.append(node[1])

					if (len(neighbor_o) > 0):
						# neighbor_o = neighbor_o[:2]
						for entity in neighbor_o:
							if ([entity, positive_sample[1], positive_sample[2]] not in raw_entities):
								command = "hpath" + " " + get_node_id(entity) + " " + get_node_id(positive_sample[1]) +  " "  +  "10000" + " " + "3" + " " + "F" + " " + "P"
								features = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()
								features =  features.split("\n")[1:-1] 

								# random_prob = []
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
											# node = "NULL"
											break # restart random walk
										else:
											random.seed(100)
											random.shuffle(relnbrs)
											if (i == 0):
												# node = "NULL"
												for tmp in relnbrs[:1000]:
													command = "ontology" + " " + get_node_id(tmp)
													node_onto = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()
													node_onto = set(node_onto.rstrip("\n").rstrip(",").split(","))
													if (len(s_onto.intersection(node_onto)) >= min(4, len(s_onto))):
														if ([tmp, positive_sample[1]] not in [x[:2] for x in negative_set]) and ([positive_sample[1], tmp] not in [x[:2] for x in negative_set]):
															if ([tmp, positive_sample[1], positive_sample[2]] not in raw_entities) and ([positive_sample[1], tmp, positive_sample[2]] not in raw_entities) and \
																	(tmp!= positive_sample[1]) and ([positive_sample[1], tmp] not in train_scenario_tocheck) and \
																		([tmp, positive_sample[1]] not in train_scenario_tocheck) :
																negative_example = [tmp, positive_sample[1], positive_sample[2], path_name, "#"]
																negative_set.append(negative_example)
																negative_cluster.append("#")
																# node = tmp
																break

											else:
												# np.random.seed(100)
												node = np.random.choice(relnbrs, 1)[0] # pick 1 nbr uniformly at random

					if (len(negative_set) >= maj_len):
						break

				i_cluster = i_cluster + 1
				if (i_cluster == len(values)):
					# i_cluster = 0
					# repeat = repeat + 1
					break
			negative_set = negative_set[:maj_len]
			negative_cluster = set(negative_cluster)

			consistency_maj.append(len(negative_cluster))

			if (len(negative_set) < maj_len):
				flag_possibility = False
				error = "Cannot create scenario: Not enough suitable negative examples.. Decrease Consistency ratio..."
	
		# np.random.seed(100)	
		while (len(negative_set) < negative_len):					
			tmp = [positive_set[np.random.randint(len(positive_set))][0], positive_set[np.random.randint(len(positive_set))][1], positive_set[0][2]]
			if (tmp not in raw_entities) and ([tmp[1], tmp[0], tmp[2]] not in raw_entities):
				tmp.append('*')
				tmp.append('*')
				if (tmp[:2] not in [x[:2] for x in negative_set]):				
					negative_set.append(tmp)
	
	output_filename = args.basedir + "splits/" +args.relation +  "/" + args.type + "_c" + str(args.consis) + "_h" + str(args.homo) + "_s" + str(args.size_scenario) + "_r" + str(args.pos_neg) + "_" + args.popular + "_scenario.tsv"
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
		print("Successfully generated !!!")
	else:
		print(error)


    
