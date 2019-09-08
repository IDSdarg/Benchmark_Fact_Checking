#   Copyright 2016 The Trustees of Indiana University.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

""" Question-answering via confusion matrix """

import sys
from contextlib import closing
import numpy as np
import os.path  
import glob
import pandas as pd
import scipy.sparse as sp
from time import time
from datetime import datetime
import traceback
from multiprocessing import Pool, cpu_count
import json

from ..utils import make_weighted, WEIGHT_FUNCTIONS
from ..algorithms.closure import epclosure, epclosuress

from scipy.sparse import SparseEfficiencyWarning
from warnings import filterwarnings
import scipy.sparse as sp

filterwarnings('ignore', category=SparseEfficiencyWarning)
import pickle

now = datetime.now
WORKER_DATA = {}
max_tasks_per_worker = 500


def populate_parser(parser):
    parser.add_argument('exp_base', metavar='base', help='experiment base directory')
    parser.add_argument('exp_spec', metavar='exp', help='experiment spec')
    # parser.add_argument('train_file', metavar='train', help='train filename')
    # parser.add_argument('test_file', metavar='test', help='test filename')

def _init_worker(A, kind):
    global WORKER_DATA
    B = A.tocsc()
    WORKER_DATA['A'] = A
    WORKER_DATA['B'] = B
    WORKER_DATA['kind'] = kind
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker(st):
    try:
        global WORKER_DATA
        A = WORKER_DATA['A']
        B = WORKER_DATA['B']
        kind = WORKER_DATA['kind']

        source = st[0]
        target = st[1]
        #targets = WORKER_DATA['targets']
        # first, compute closure to all targets
        #D, _ = epclosuress(A, source, B=B, kind=kind)
        #D = D[targets]
        # then, check if any element needs to be recomputed without its edge
        # removal
        #idx, = np.where(D == 1.0)  # direct neighbors, by definition
        #for i in idx:
        #    target = targets[i]
        s = A[source, target]
        A[source, target] = 0.
        B[source, target] = 0.
        A.eliminate_zeros()
        B.eliminate_zeros()
        d, _ = epclosure(A, source, target, B=B, kind=kind)
        A[source, target] = s
        B[source, target] = s
        return d
    except Exception:
        val, ty, tb = sys.exc_info()
        traceback.print_tb(tb)
        raise


def confmatrix(A, sources_targets, nprocs=None, kind='ultrametric'):
    """
    Confusion matrix with edge removal

    Parameters
    ----------
    A : array_like
        Adjacency matrix.

    sources : array_like
        Source node IDs

    targets : array_like or dict or list of lists
        Target node IDs

    nprocs : int
        The number of processes to use. Default: 90% of the available CPUs or
        2, whichever the largest.

    kind : string
        The metric type. See `knowledge_linker.closure.epclosuress`.

    Returns
    -------

    simil : array_like
        Array of similarity values

    """
    if nprocs is None:
        # by default use 90% of available processors or 2, whichever the
        # largest.
        nprocs = max(int(0.9 * cpu_count()), 2)
    # allocate array to be passed as shared memory
    print >> sys.stderr, \
        '{}: copying graph data to shared memory.'.format(now())
    A = sp.csr_matrix(A)

    initargs = (A, kind)
    print >> sys.stderr, \
        '{}: launching pool of {} workers.'.format(now(), nprocs)
    pool = Pool(processes=nprocs, initializer=_init_worker,
                initargs=initargs, maxtasksperchild=max_tasks_per_worker)
    try:
        with closing(pool):
            result = pool.map_async(_worker, sources_targets)
            while not result.ready():
                result.wait(1)
        pool.join()
        if result.successful():
            print >> sys.stderr, '{}: done'.format(now())
            return result.get()
        else:
            print >> sys.stderr, "{}: There was an error in " \
                "the pool.".format(now())
            sys.exit(2)  # ERROR occurred
    except KeyboardInterrupt:
        print "^C"
        pool.terminate()
        sys.exit(1)  # SIGINT received

def getKey(item):
    return item[1]

def main(ns):
    
    spec = json.load(open(ns.exp_base + "experiment_specs/" + ns.exp_spec))
    nodespath = ns.exp_base + "graphs/node_dict.tsv"
    edgetype = ns.exp_base + "graphs/edge_dict.tsv"
    
    edgespath = ns.exp_base + "graphs/graph_chi/edges.tsv"
    closure = spec['operation']['features']['closure']
    weight = spec['operation']['features']['weight']
    # input = ns.exp_base + 'splits/' + spec['split']['name'] + '/' + spec['split']['level']
    is_directed = spec['operation']['features']['is_directed']

    adjpath = ns.exp_base + "graphs/adj_" + weight + "_" + is_directed + "_.npz"

    nprocs = spec['operation']['features']['nprocs']
    ## load nodes list

    test_file = spec["split"]["test_file"]
    train_file = spec["split"]["train_file"]
    
    print >> sys.stderr, '{}: reading nodes..'.format(now())

    num_entities = 0
    with closing(open(nodespath)) as f:
        nodes = f.readlines()
        num_entities = len(nodes)
        node_dict = [nodes[i].rstrip('\n').split('\t') for i in range(len(nodes))]
        
        vertexmap = dict([[entity, int(i)-1] for i, entity in node_dict])
        del node_dict, nodes
    
    import os.path
    if os.path.isfile(adjpath):
        print >> sys.stderr, '{}: reading graph..'.format(now())
        A = sp.load_npz(adjpath)
    else:
        print >> sys.stderr, '{}: creating graph..'.format(now())

        from itertools import groupby
        from operator import itemgetter
        from ..utils import coo_dtype
        from tqdm import tqdm
        data = []
        with closing(open(edgespath)) as f:
            for labeled_edge in f:     
                out_entity, in_entity, rel_ = labeled_edge.rstrip('\n').split('\t')    
                data.append((int(out_entity)-1, int(in_entity)-1, 1.0))

        #del edge_list, labeled_edge
        data = np.asarray(data, dtype=coo_dtype)
        A = make_weighted(data, num_entities, undirected= (False if (is_directed == "true") else True), weight= weight)
        sp.save_npz(adjpath, A)
        print >> sys.stderr, 'info: adj written to {}'.format(adjpath)

    
        
    
    with closing(open(test_file)) as f:
        testing_set = f.readlines()
        testing_set = testing_set[1:]
        testing_set = [testing_set[i].rstrip('\n').split('\t')[:4] for i in range(len(testing_set))]       
        testing_set_id = [[vertexmap[i], vertexmap[j]] for i, j,_,_ in testing_set]  

    for source, target in testing_set_id:
        A[source, target] = 0.
        A[target, source] = 0.
        A.eliminate_zeros()   

    t1 = time()
    B = confmatrix(A, testing_set_id, kind=closure)
    print 'Time taken: {:.2f}s\n'.format(time() - t1)
    outf = pd.concat([pd.DataFrame(testing_set,columns=['s', 'o', 'p', 'true_label']), pd.DataFrame(B, columns=['predict_proba'])], axis=1)
    
    outdirs = test_file.replace("scenario", "klinker_score")
    if not os.path.exists(os.path.dirname(outdirs)):
        os.makedirs(os.path.dirname(outdirs))

    outf.to_csv(outdirs, sep='\t', index=False, encoding='utf-8')
    print '* Saved score results'  


