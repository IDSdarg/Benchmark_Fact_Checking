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

def find_best_model(X, y, scoring='roc_auc', cv=10):
	steps = [('clf', LogisticRegression())]
	pipe = Pipeline(steps)
	params = {'clf__C': [1, 5, 10, 15, 20]}
	grid_search = GridSearchCV(pipe, param_grid=params, cv=cv, refit=True, scoring=scoring)
	grid_search.fit(X, y)
	best = {
		'clf': grid_search.best_estimator_, 
		'best_score': grid_search.best_score_,
		'best_param': grid_search.best_params_
	}
	return best

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
	# parser.add_argument('train_file', metavar='train', help='train filename')
	# parser.add_argument('test_file', metavar='test', help='test filename')
	

	args = parser.parse_args()

	spec = json.load(open(args.exp_base + "experiment_specs/" + args.exp_spec))
	# nodespath = args.exp_base + "graphs/node_dict.tsv"
	edgetype = args.exp_base + "graphs/edge_dict.tsv"
	edgespath = args.exp_base + "graphs/graph_chi/edges.tsv"
	max_depth = spec['operation']['features']['max_depth']
	test_file = spec["split"]["test_file"]
	train_file = spec["split"]["train_file"]

	is_directed = spec['operation']['features']['is_directed']
	nprocs = spec['operation']['features']['nprocs']

	# with closing(open(nodespath)) as f:
	# 	nodes = f.readlines()
	# 	node_dict = [nodes[i].rstrip('\n').split('\t') for i in range(len(nodes))]
	# 	from collections import OrderedDict
	# 	nodemap = OrderedDict(( (entity, i) for i, entity in sorted(node_dict, key=getKey) ))
	# 	del node_dict, nodes

	
	with closing(open(edgetype)) as f:
		nodes = f.readlines()
	edge_dict = [nodes[i].rstrip('\n').split('\t') for i in range(len(nodes))]
	vertexmap = OrderedDict(( (int(i), edge) for i, edge in sorted(edge_dict, key=getKey) ))
	edge_dict = dict([[edge, i] for i, edge in edge_dict])


	with closing(open(train_file)) as f:
		train_set = f.readlines()
		train_set = train_set[1:]
		train_set = [train_set[i].rstrip('\n').split('\t')[:4] for i in range(len(train_set))]
		# testing_set_id = [[nodemap[i[0]], nodemap[i[1]], i[3]] for i in testing_set]
		train_set_id = [[get_node_id(i[0]), get_node_id(i[1]), edge_dict[i[2]], i[3]] for i in train_set]

	commands = [[int(spo[3]), "hpath" + " " + spo[0] + " " + spo[1] + " "  + spo[2] + " " + str(max_depth) + " " + ("T" if (is_directed == "true") else "F") + " " + "P"] for spo in train_set_id]
	p = multiprocessing.Pool(nprocs)
	paths = p.map(server_connect, commands)
	#paths = [for label, command in commands]

	paths =  [[label, path.split("\n")[1:-1]] for label, path in paths]
	paths = [[label, {x :path.count(x) for x in path}] for label, path in paths]

	df_train_paths = pd.DataFrame([path for label, path in paths])
	df_train_paths.rename(columns=lambda x: map_predicate(x), inplace=True)
	df_train_paths.index = [0]*len(df_train_paths)

	labels_train = np.array([1 if label == 1 else 0 for label, path in paths]).astype('int')
	
	with closing(open(test_file)) as f:
		test_set = f.readlines()
		test_set = test_set[1:]
		test_set = [test_set[i].rstrip('\n').split('\t')[:4] for i in range(len(test_set))]
		# testing_set_id = [[nodemap[i[0]], nodemap[i[1]], i[3]] for i in testing_set]
		test_set_id = [[get_node_id(i[0]), get_node_id(i[1]), edge_dict[i[2]], i[3]] for i in test_set]

	commands = [[int(spo[3]), "hpath" + " " + spo[0] + " " + spo[1] + " "  + spo[2] + " " + str(max_depth) + " " + ("T" if (is_directed == "true") else "F") + " " + "P"] for spo in test_set_id]
	p = multiprocessing.Pool(nprocs)
	paths = p.map(server_connect, commands)
	#paths = [for label, command in commands]

	paths =  [[label, path.split("\n")[1:-1]] for label, path in paths]
	paths = [[label, {x :path.count(x) for x in path}] for label, path in paths]

	df_test_paths = pd.DataFrame([path for label, path in paths])
	df_test_paths.rename(columns=lambda x: map_predicate(x), inplace=True)
	df_test_paths.index = [1]*len(df_test_paths)

	labels_test = np.array([1 if label == 1 else 0 for label, path in paths]).astype('int')
	
	
	df_predicate_full = pd.concat([df_train_paths, df_test_paths])
	df_predicate_full = df_predicate_full.fillna(0)

	test_predicates = df_predicate_full.loc[[1]].as_matrix().astype('float')
	train_predicates = df_predicate_full.loc[[0]].as_matrix().astype('float')

	model = find_best_model(train_predicates, labels_train, cv=10)

	kg_pred_prob = model['clf'].predict_proba(test_predicates)[:,1]
	# kg_pred_label = model['clf'].predict(test_predicates)

	coef_ = model['clf'].named_steps['clf'].coef_

	feature_name = df_predicate_full.loc[[0]].columns.values

	outf = pd.concat([pd.DataFrame(test_set, columns=['s', 'o', 'p', 'true_label']), pd.DataFrame(kg_pred_prob, columns=['predict_proba'])], axis=1)
	outdirs = test_file.replace("scenario", "kgminer_score")
	if not os.path.exists(os.path.dirname(outdirs)):
		os.makedirs(os.path.dirname(outdirs))
	outf.to_csv(outdirs, sep='\t', index=False, encoding='utf-8')
	#print '* Saved score results'

	outdirs = test_file.replace("scenario", "kgminer_weight")
	if not os.path.exists(os.path.dirname(outdirs)):
		os.makedirs(os.path.dirname(outdirs))		

	outf = pd.DataFrame({'feature': feature_name, 'weight': coef_[0]})
	outf.to_csv(outdirs, sep='\t', index=False, encoding='utf-8')
	#print '* Saved weight results'	


