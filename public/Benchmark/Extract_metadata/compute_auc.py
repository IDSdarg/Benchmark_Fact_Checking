
import sys
from contextlib import closing
import argparse
import traceback
import os
import errno
import numpy as np
import glob
from sklearn import metrics
import json
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':

	parser = argparse.ArgumentParser(
	    description=__doc__, 
	    formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	parser.add_argument('score_file', metavar='test', help='score filename')

	args = parser.parse_args()
		
	file_ = open(args.score_file, 'r')

	scores_ = file_.readlines()
	scores_ = scores_[1:]
	scores_ = [score_.rstrip('\n').split('\t') for score_ in scores_]
	file_.close()
		    
	pred_prob = [float(x[4]) for x in scores_]
	true_label = [int(x[3]) for x in scores_]
	# print "--- AUROC: " + str(metrics.roc_auc_score(sfe_true_label, sfe_pred_prob)) + "-----"
	# plt.figure()
	lw = 2
	fpr, tpr, _ = roc_curve(true_label, pred_prob)
	plt.plot(fpr, tpr, color='darkorange',
	         lw=lw)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.tick_params(axis='both',  labelsize=15)
	plt.xlabel('False Positive Rate', {'fontsize': 20})
	plt.ylabel('True Positive Rate', {'fontsize': 20})
	plt.title('Receiver operating characteristic', fontsize=20)
	# plt.legend(loc="lower right")
	plt.savefig('../../roc_curve.svg', bbox_inches='tight', dpi=1500)
	print(str(round(metrics.roc_auc_score(true_label, pred_prob),2)))

