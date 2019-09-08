import sys
from contextlib import closing
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from datetime import datetime
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

if __name__ == '__main__':

	parser = argparse.ArgumentParser(
        description=__doc__, 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
	parser.add_argument('basedir', metavar='base', help='experiment base directory')
	parser.add_argument('-r', type=str, required=True,
            dest='relation', help='Relation')

	args = parser.parse_args()

	with closing(open(args.basedir + "graphs/edge_dict.tsv")) as f:
		nodes = f.readlines()
		edge_types = [nodes[i].rstrip('\n').split('\t') for i in range(len(nodes))]
		vertexmap = OrderedDict(( (int(i), edge) for i, edge in sorted(edge_types, key=getKey) ))
		del edge_types, nodes

	if not os.path.isfile(args.basedir + "splits/" + args.relation + "/" + args.relation + ".tsv"):
		command = "getpairsbyrel" + " " + args.relation + " " + str(1.0)
		pairs_by_rel = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()
		# pairs_by_rel = pairs_by_rel.split('\n')[1:-1][:4000]
		pairs_by_rel = pairs_by_rel.split('\n')[1:-1]
		pairs_by_rel = [pair.split('\t') for pair in pairs_by_rel]
		pairs_by_rel = [[pair[0], pair[1], pair[2], pair[3], pair[4], int(pair[5]), int(pair[6])] for pair in pairs_by_rel]
		
		if (len(pairs_by_rel) > 1000):
			try:
				os.makedirs(os.path.dirname(args.basedir + "splits/" + args.relation + "/" + args.relation + ".tsv"))
			except OSError as exc: # Guard against race condition:
				if exc.errno != errno.EEXIST:
					raise		
			try:
				os.makedirs(os.path.dirname(args.basedir + "splits/" + args.relation + "/" + args.relation + "_id.tsv"))
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise
			ids =  open(args.basedir + "splits/" +  args.relation + "/" + args.relation + ".tsv", 'w')
			ids_2 =  open(args.basedir + "splits/" + args.relation + "/" + args.relation + "_id.tsv", 'w')	
			pairs_by_rel = sorted(pairs_by_rel, key=operator.itemgetter(5), reverse = True)		
			for pair in pairs_by_rel:
				ids.write(pair[3] + '\t' + pair[4] + '\t' + args.relation + '\t' + str(pair[5]) + '\t' + str(pair[6]) +'\n')
				ids_2.write(pair[0] + '\t' + pair[1] + '\t' + pair[2] + '\t' + str(pair[5]) + '\t' + str(pair[6]) +'\n')
			ids.close()
			ids_2.close()
			print("Suitable predicate: enough triples !")
		else:
			print("Not enough triples for this predicate. Change to others !")
	else:
		print("Suitable predicate: enough triples !")

