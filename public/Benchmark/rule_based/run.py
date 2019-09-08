import os
import argparse
import numpy as np
from sklearn import metrics
import json
import pandas as pd
import random
from contextlib import closing
from collections import OrderedDict
import pickle

def should_keep(x, ex_list):
    if x["computedConfidence"]>0.5:
        for s in ex_list:
            if s in x["premise"]:
                return False
        return True
    else:
        return False


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
		# nodespath = args.exp_base + "graphs/node_dict.tsv"
	edgetype = args.exp_base + "graphs/edge_dict.tsv"
	edgespath = args.exp_base + "graphs/graph_chi/edges.tsv"
	relation = spec["split"]["name"]

	test_file = spec["split"]["test_file"]

	ex_list = [">", ">=", "<", "<="]
	ex_list.append(relation)

	with closing(open(args.exp_base + "graphs/edge_dict.tsv")) as f:
		edge_types = f.readlines()
		edge_types = [edge.rstrip('\n').split('\t')[1] for edge in edge_types]

	rel_set = []
	neg_rules = []
	pos_rules = []
	try:	
		with open(args.exp_base + "splits/" + relation + "/" + "negative_rule.json") as json_file:
			negative_data = json.load(json_file)
			negative_data = [x for x in negative_data if should_keep(x, ex_list)]
			for x in negative_data:
				neg_rules.append([x["premiseTriples"], 1.0-x["computedConfidence"]])
			   	for rel in x["premiseTriples"]:
			        	rel_set.append(rel["predicate"].split("/")[-1])


	
		with open(args.exp_base + "splits/" + relation + "/" + "positive_rule.json") as json_file:
			positive_data = json.load(json_file)
			positive_data = [x for x in positive_data if should_keep(x, ex_list)]
			for x in positive_data:
				pos_rules.append([x["premiseTriples"], x["computedConfidence"]])
			    	for rel in x["premiseTriples"]:
	
			        	rel_set.append(rel["predicate"].split("/")[-1])

	except:
		print("dkm")
		exit()
	rel_set = set(rel_set)
	print(len(rel_set))
	coverage_rel = {}

	if not os.path.isfile(args.exp_base + "splits/" + relation + "/" + "rule_coverage.p"):
		count = 0
		for rel in rel_set:
			print(str(count) + " -- " + rel)
			count+=1
			if rel in edge_types:
				command = "getpairsbyrel" + " " + rel + " " + str(1.0)
				pairs_by_rel = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()
				# pairs_by_rel = pairs_by_rel.split('\n')[1:-1][:4000]
				pairs_by_rel = pairs_by_rel.split('\n')[1:-1]
				pairs_by_rel = [pair.split('\t') for pair in pairs_by_rel]
				coverage_rel[rel] = [set([pair[0] for pair in pairs_by_rel]),  set([pair[1] for pair in pairs_by_rel])]
			else:
				coverage_rel[rel] = [set([]), set([])]

		pickle.dump( coverage_rel, open( args.exp_base + "splits/" + relation + "/" + "rule_coverage.p", "wb" ) )

	else:
		coverage_rel = pickle.load(open(args.exp_base + "splits/" + relation + "/" + "rule_coverage.p",  "rb"))
	
	
	with closing(open(test_file)) as f:
		test_set = f.readlines()
		test_set = test_set[1:]
		test_set = [test_set[i].rstrip('\n').split('\t')[:4] for i in range(len(test_set))]
		# testing_set_id = [[nodemap[i[0]], nodemap[i[1]], i[3]] for i in testing_set]
		test_set_id = [[get_node_id(i[0]), get_node_id(i[1]),  i[3]] for i in test_set]

	scores = []
	index = 0
	for sub_spl, obj_spl, label in test_set_id:
		print(index)
		index += 1
		scores_tmp = []
		for pos_rl, conf_rl in pos_rules:
			atoms = {}
			for tpl in pos_rl:
				pred = tpl["predicate"].split("/")[-1]
				if pred != "!=":
					try:
						atoms[tpl["subject"]].append(coverage_rel[pred][0])
					except:
						atoms[tpl["subject"]] = []
						atoms[tpl["subject"]].append(coverage_rel[pred][0])
					try:
						atoms[tpl["object"]].append(coverage_rel[pred][1])
					except:
						atoms[tpl["object"]] = []
						atoms[tpl["object"]].append(coverage_rel[pred][1])
				else:
					try:
						atoms[pred].append([tpl["subject"], tpl["object"]])
					except:
						atoms[pred] = []
						atoms[pred].append([tpl["subject"], tpl["object"]])



			keep_rl = True
			for atom in atoms:
				if atom == "subject":
					if all([sub_spl in x for x in atoms[atom]]):
						pass
					else:
						keep_rl = False
						break

				if atom == "object":
					if all([obj_spl in x for x in atoms[atom]]):
						pass
					else:
						keep_rl = False
						break	

				if atom == "!=":
					flag_in = True
					for pair in atoms[atom]:
						x1 = set.intersection(*atoms[pair[0]])
						x2 = set.intersection(*atoms[pair[1]])
						if x1 & x2:
							flag_in = False
							break
					if flag_in:
						pass
					else:
						keep_rl = False
						break

				if (atom != "subject") and (atom != "object") and (atom != "!="):
					if set.intersection(*atoms[atom]):
						pass
					else:
						keep_rl = False
						break
		

			if keep_rl:
				scores_tmp.append([pos_rl, conf_rl])
				break

		for pos_rl, conf_rl in neg_rules:
			atoms = {}
			for tpl in pos_rl:
				pred = tpl["predicate"].split("/")[-1]
				if pred != "!=":
					try:
						atoms[tpl["subject"]].append(coverage_rel[pred][0])
					except:
						atoms[tpl["subject"]] = []
						atoms[tpl["subject"]].append(coverage_rel[pred][0])
					try:
						atoms[tpl["object"]].append(coverage_rel[pred][1])
					except:
						atoms[tpl["object"]] = []
						atoms[tpl["object"]].append(coverage_rel[pred][1])
				else:
					try:
						atoms[pred].append([tpl["subject"], tpl["object"]])
					except:
						atoms[pred] = []
						atoms[pred].append([tpl["subject"], tpl["object"]])

			keep_rl = True
			for atom in atoms:
				if atom == "subject":
					if all([sub_spl in x for x in atoms[atom]]):
						pass
					else:
						keep_rl = False
						break

				if atom == "object":
					if all([obj_spl in x for x in atoms[atom]]):
						pass
					else:
						keep_rl = False
						break	

				if atom == "!=":
					flag_in = True
					for pair in atoms[atom]:
						x1 = set.intersection(*atoms[pair[0]])
						x2 = set.intersection(*atoms[pair[1]])
						if x1 & x2:
							flag_in = False
							break
					if flag_in:
						pass
					else:
						keep_rl = False
						break

				if (atom != "subject") and (atom != "object") and (atom != "!="):
					if set.intersection(*atoms[atom]):
						pass
					else:
						keep_rl = False
						break
		

			if keep_rl:
				scores_tmp.append([pos_rl, conf_rl])
				break
		prob = [x[1] for x in scores_tmp]
		if prob:
			prob = np.mean(prob)
		else:
			prob = 0.5
		if label == "1":
			label = 1
		else:
			label = 0
		scores.append([prob, label])
		
	print(scores)
	print(str(round(metrics.roc_auc_score([x[1] for x in scores], [x[0] for x in scores]),2)))



	predict_proba = np.array([score[0] for score in scores])


	outf = pd.concat([pd.DataFrame(test_set, columns=['s', 'o', 'p', 'true_label']), pd.DataFrame(predict_proba, columns=['predict_proba'])], axis=1)
	outdirs = test_file.replace("scenario", "rule_score")
	if not os.path.exists(os.path.dirname(outdirs)):
		os.makedirs(os.path.dirname(outdirs))
	outf.to_csv(outdirs, sep='\t', index=False, encoding='utf-8')
	#print '* Saved score results'
