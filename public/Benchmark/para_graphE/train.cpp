/**
	* Copyright (c) 2016 LIBBLE team supervised by Dr. Wu-Jun LI at Nanjing University.
	* All Rights Reserved.
	* Licensed under the Apache License, Version 2.0 (the "License");
	* you may not use this file except in compliance with the License.
	* You may obtain a copy of the License at
	*
	* http://www.apache.org/licenses/LICENSE-2.0
	*
	* Unless required by applicable law or agreed to in writing, software
	* distributed under the License is distributed on an "AS IS" BASIS,
	* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	* See the License for the specific language governing permissions and
	* limitations under the License. */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <thread>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <tuple>
#include <set>
using namespace std;

#include "math_func.hpp"
#include "transbase.hpp"
#include "transe.hpp"
#include "json.hpp"


using json = nlohmann::json;
//parameters of program
int nthreads = 4;	// number of threads
string data_path = "";	//input folder of the kg
string exclude_relation_file = "";
vector<double> epoch_loss;
transbase *trans_ptr = nullptr;
string method = "TransE";
string relation="";
int use_tmp = 0;	//use a tmp value in each batch 

//hyper-parameters of algorithm
int embedding_dim = 50;	//the dimension of embeddings
int dim2 = 50;	//TransR only
double learning_rate = 0.01;  //learning rate
int corr_method = 0;	//sampling method, 0 = uniform, 1 = bernoulli 
double margin = 1;		//the margin of pairwise ranking loss 
int nepoches = 1000; 	//the num of epoches in training
int nbatches = 1;		//num of batches per epoch
int l1_norm = 1;		//0 = l2-norm, 1 = l1-norm
double orth_value = 0.1;	//TransH only

//parameters of the knowledge graph
int entity_num = 0, relation_num = 0 ; // initial value
int train_num = 0, valid_num = 0;
int exclude_relation = 0;
//data structure of algorithm
vector<int> train_h, train_r, train_t;
vector<int> valid_h, valid_r, valid_t; 
vector<set<int> > hset, tset;	//hset = <r, {h | <h,r,t> \in train_set}>
vector<int> r_count;	//count the times a relation appears
vector<double> tph, hpt;	
set<tuple<int, int, int> > triple_count;	

int arg_handler(string str, int argc, char **argv) {
	int pos;
	for (pos = 0; pos < argc; pos++) {
		if (str.compare(argv[pos]) == 0) {
			return pos;
		}
	}
}

void read_input() {
	int h, r, t;
	string tmp;
	int num_exlude = 0;
	ifstream file;

	if (corr_method == 1) {
		hset.resize(relation_num);
		tset.resize(relation_num);
		tph.resize(relation_num);
		hpt.resize(relation_num);
		r_count.resize(relation_num);
	}
	
	map<string, int> entity_dict;

	file.open(data_path + "graphs/node_dict.tsv");
	
	while(!file.eof()) {
		file >> h ;
		if( file.eof() ) break;
		file.get(); 
		file >> tmp ;
		entity_dict[tmp] = h;
		entity_num = entity_num + 1;	
	}
	file.close();

	cout << "Number of entity: " << entity_num <<  endl;

	map<string, int> relation_dict;

	file.open(data_path + "graphs/edge_dict.tsv");
	
	while(!file.eof()) {
		file >> h ;
		if( file.eof() ) break;
		file.get(); 
		file >> tmp ;
		relation_dict[tmp] = h;
		relation_num = relation_num + 1;	
	}
	file.close();
	cout << "Number of relation type: " << relation_num <<  endl;

	vector<int> check_h1, check_t1, check_r1;
	string h1, t1, r1, l1, l11, l12;
	file.open(exclude_relation_file);
	if (file.is_open())
	{
		file >> h1 >> t1 >> r1 >> l1 >> l11 >> l12;
		while(!file.eof()) {
		// while (num_exlude<5000) {
			file >> h1 >> t1 >> r1 >> l1 >> l11 >> l12;
			num_exlude =  num_exlude + 1;
			check_h1.push_back(entity_dict[h1]);
			check_t1.push_back(entity_dict[t1]);
		    if (find(check_r1.begin(), check_r1.end(), relation_dict[r1]) == check_r1.end()) { 
		        check_r1.push_back(relation_dict[r1]); 
		    }
			
		}

		// exclude_relation = relation_dict[r1];
		file.close();
		cout << "Number of exclude examples: " << num_exlude <<  endl;
		cout << "Number of exclude relations: " << check_r1.size() <<  endl;
	}
	else
	{
		cout << "No exclude relation: " <<  endl;
	}


	file.open(data_path + "graphs/graph_chi/edges.tsv");

	
	while(!file.eof()) {
		file >> h ;
		if( file.eof() ) break;
		file.get();
		file >> t ;
		file.get();
		file >> r;

		int flag = 1;
		if (num_exlude > 0) {
			if (find(check_r1.begin(), check_r1.end(), r) != check_r1.end()) { 
				for (int j=0; j<num_exlude; j++) {
					if ((h == check_h1[j]) && (t == check_t1[j])) {
						flag = 0;
						break;
					}
				}
			}
		}

		if (flag != 0) {
			train_h.push_back(h-1);
			train_r.push_back(r-1);
			train_t.push_back(t-1);
			if (corr_method==1) {
				hset[r-1].insert(h-1);
				tset[r-1].insert(t-1);
				r_count[r-1]++;
			}
			triple_count.insert(make_tuple(h-1,r-1,t-1));
			train_num = train_num + 1;
		}
	}
	file.close();

	cout << "Number of training example: " << train_num <<  endl;
	
	if (corr_method==1) {
		for (int r=0; r<relation_num; r++) {
			tph[r] = (double)r_count[r] / hset[r].size();
			hpt[r] = (double)r_count[r] / tset[r].size();
		}
	}

}

double valid_loss() {		//compute the loss on the valid set
	double total_loss = 0;
	for (int i=0; i<valid_num; i++) {
		total_loss += trans_ptr->triple_loss(valid_h[i], valid_r[i], valid_t[i]);
	}
	return total_loss;
}

double train_loss() {	//compute the loss on the train set
	double total_loss = 0;
	for (int i=0; i<train_num; i++) {
		total_loss += trans_ptr->triple_loss(train_h[i], train_r[i], train_t[i]);
	}
	return total_loss;
}

void pairwise_update(int r, int h, int t, int h_, int t_, int id) {	
	double gold_loss = trans_ptr->triple_loss(h, r, t), cor_loss = trans_ptr->triple_loss(h_, r, t_);
	double temp = gold_loss + margin - cor_loss;
	if (temp > 0) {
		epoch_loss[id] += temp;
		trans_ptr->gradient(r, h, t, h_, t_);
	}
}
void rand_train(int id, int size) {	//optimize using sgd without considering overlapping problem
	default_random_engine generator(random_device{}());
	uniform_int_distribution<int> triple_unif(0, train_num-1);
	uniform_int_distribution<int> entity_unif(0, entity_num-1);
	uniform_real_distribution<double> unif(-1, 1);
	for (int i=0; i<size; i++) {
		int tri_id = triple_unif(generator);
		int h = train_h[tri_id], r = train_r[tri_id], t = train_t[tri_id];
		int cor_id;	//the id of sub entity
		double prob;
		
		if (corr_method == 1) {
			uniform_real_distribution<double> bern(-hpt[r] ,tph[r]);
			prob = bern(generator);
		} else {
			prob = unif(generator);
		}
				
		if (prob > 0) {	//change head
			cor_id = entity_unif(generator);
			while (triple_count.find(make_tuple(cor_id,r,t)) != triple_count.end())
				cor_id = entity_unif(generator);
			pairwise_update(r, h, t, cor_id, t, id);
		} else {	//change tail
			cor_id = entity_unif(generator);
			while (triple_count.find(make_tuple(h,r,cor_id)) != triple_count.end())
				cor_id = entity_unif(generator);
			pairwise_update(r, h, t, h, cor_id, id);
		}
	}
}

void method_ptr_binding(string method) {
	if (method.compare("TransE") == 0) 
		trans_ptr = new transE(entity_num, relation_num, embedding_dim, l1_norm, learning_rate, use_tmp);
	else {
		cout << "no such method!" << endl;
		exit(1);
	}
}

int main(int argc, char **argv) {

	data_path = string(argv[1]);

	std::ifstream i(data_path + "experiment_specs/" + string(argv[2]));
	json spec;
	i >> spec;

	method = spec["operation"]["method"];
	embedding_dim = spec["operation"]["features"]["embed_dim"];
	learning_rate = spec["operation"]["features"]["learning_rate"];
	nepoches = spec["operation"]["features"]["nepoches"];
	nthreads = spec["operation"]["features"]["nprocs"];
	relation = spec["split"]["name"];
	exclude_relation_file = spec["split"]["test_file"];

	cout << "args settings:" << endl
		<< "----------" << endl
		<< "method " << method << endl
		<< "thread number " << nthreads << endl
		<< "dimension "  << embedding_dim << endl
		<< "dimension2 (only in transR) " << dim2 << endl
		<< "orthogonal value (only in transH) " << orth_value << endl
		<< "learning rate " << learning_rate << endl
		<< "sampling corrupt triple method " << (corr_method == 0 ? "unif" : "bern") << endl
		<< "margin " << margin << endl
		<< "epoch number " << nepoches << endl
		<< "batch number " << nbatches << endl
		<< "norm " << (l1_norm == 0 ? "L2" : "L1") << endl
		<< "use tmp value in batches " << (use_tmp == 1 ? "Yes" : "No") << endl
		<< "data path " << data_path << endl
		<< "scenario path " << exclude_relation_file << endl
		<< "----------" << endl;
	
	this_thread::sleep_for(chrono::seconds(3));
	
	//initializing
	
	read_input();
	
	method_ptr_binding(method);
	

	trans_ptr->initial();
	
	int batch_size = train_num / nbatches;
	int thread_size = batch_size / nthreads;	//number of triples a thread handles in one run
	cout << "initializing process done." << endl;
	
	//set-up multi-thread and start learning
	thread workers[nthreads];
	epoch_loss.resize(nthreads);

	auto start = chrono::high_resolution_clock::now();
	for (int epoch = 0; epoch < nepoches; epoch++) {	
		reset(epoch_loss);
		for (int batch = 0; batch < nbatches; batch++) {
			for (int id=0; id<nthreads; id++) {
				workers[id] = thread(rand_train, id, thread_size);
			}
			for (auto &x: workers)
				x.join();
			if (use_tmp)	//if tmp is used then update for batch is needed
				trans_ptr->batch_update();
		} //batch iteration
		
		cout << "loss of epoch " << epoch << " is " << sum(epoch_loss) << endl;
	} //epoch iteration
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double> diff = end-start;
	cout << "training process done, total time: " << diff.count() << " s." << endl;
	
	//finalizing
	trans_ptr->save_to_file(data_path+"graphs/", relation);
	cout << "the embeddings are already saved in files." << endl;
}
