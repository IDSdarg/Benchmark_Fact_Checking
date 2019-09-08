import numpy as np
import pandas as pd
import glob
import os
import argparse
import json
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from contextlib import closing
from collections import OrderedDict
import multiprocessing



def getKey(item):
    return item[1]


vertexmap = OrderedDict()
def map_predicate(s):
	s = s.rstrip(',').split(',')
	s = [ ('(-1)' if (int(x) < 0) else '') + vertexmap[abs(int(x))] for x in s]
	return ','.join(s)

def server_connect(command):
	return [command[0], os.popen("echo \"" + command[1] +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()] 

def get_node_id(value):
	command = "node_id" + " " + value
	return os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()
if __name__ == '__main__':
	parser = argparse.ArgumentParser(
        description=__doc__, 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

	parser.add_argument('exp_base', metavar='base', help='experiment base directory')
	parser.add_argument('exp_spec', metavar='exp', help='experiment spec')

	args = parser.parse_args()

	spec = json.load(open(args.exp_base + "experiment_specs/" + args.exp_spec))

	edgetype = args.exp_base + "graphs/edge_dict.tsv"
		
	method = spec["operation"]["method"]
	embedding_dim = spec["operation"]["features"]["embed_dim"]

	nthreads = spec["operation"]["features"]["nprocs"];
	test_file = spec["split"]["test_file"];
	relation = spec["split"]["name"]

	print "Loading embedding..."
	with closing(open(edgetype)) as f:
		nodes = f.readlines()
	edge_dict = [nodes[i].rstrip('\n').split('\t') for i in range(len(nodes))]
	vertexmap = OrderedDict(( (int(i), edge) for i, edge in sorted(edge_dict, key=getKey) ))
	edge_dict = dict([[edge, i] for i, edge in edge_dict])

	with closing(open(args.exp_base + "graphs/" +  "Full_TransE_entity_vec.txt")) as f:
		entity_vec = f.readlines()
	entity_vec = entity_vec[1:]
	# entity_vec = [node.rstrip('\n').split(' ')[:-1] for node in nodes[1:]]

	with closing(open(args.exp_base + "graphs/" +  "Full_TransE_relation_vec.txt")) as f:
		nodes = f.readlines()
	relation_vec = [node.rstrip('\n').split(' ')[:-1] for node in nodes[1:]]
	
	del nodes 
	print "Calculating test score ..."
	with closing(open(test_file)) as f:
		test_set = f.readlines()
		test_set = test_set[1:]
		test_set = [test_set[i].rstrip('\n').split('\t')[:4] for i in range(len(test_set))]
		# testing_set_id = [[nodemap[i[0]], nodemap[i[1]], i[3]] for i in testing_set]
		test_set_id = [[get_node_id(i[0]), get_node_id(i[1]), edge_dict[i[2]], i[3]] for i in test_set]

	predict_proba = []
	for test_pair in test_set_id:
		h = int(test_pair[0])-1
		r = int(test_pair[2])-1
		t = int(test_pair[1])-1		
		loss = np.array(map(float, entity_vec[h].rstrip('\n').split(' ')[:-1])) + np.array(map(float, relation_vec[r])) - np.array(map(float, entity_vec[t].rstrip('\n').split(' ')[:-1]))
		predict_proba.append(np.abs(loss).sum())
		
		
	predict_proba = np.array(predict_proba)
	predict_proba = 1 - (predict_proba - min(predict_proba))/(max(predict_proba)-min(predict_proba))


	outf = pd.concat([pd.DataFrame(test_set, columns=['s', 'o', 'p', 'true_label']), pd.DataFrame(predict_proba, columns=['predict_proba'])], axis=1)
	outdirs = test_file.replace("scenario", "transe_score")
	if not os.path.exists(os.path.dirname(outdirs)):
		os.makedirs(os.path.dirname(outdirs))
	outf.to_csv(outdirs, sep='\t', index=False, encoding='utf-8')
	#print '* Saved score results'

	


