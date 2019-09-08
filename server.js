const express = require('express')
const app = express()
const bodyParser = require('body-parser');
const shell = require('shelljs');
var path = require('path');


app.use(express.static('public'));
app.set('view engine', 'ejs')
app.set('view options', { pretty: true });
app.use(bodyParser.urlencoded({ extended: true }));
// var timeout = require('connect-timeout');
// app.use(timeout('10000s'));
const fs = require('fs');



app.get('/', function (req, res) {
	res.render('index');
})

app.post('/kb_process', function (req, res) {
  	console.log( " KB Processing");
	let kb = req.body.kb;
	if (kb == "dbpedia") {
		fs.readFile(path.join(__dirname + "/public/" +"/Benchmark/data/dbpedia/graphs/edge_dict.tsv"), function (err, data) {
			if (err) {
			   throw err; 
			}
			let  predicate = data.toString().split('\n').map(item => item.split('\t')[1]).sort();

			res.send(predicate);
		});
	}
  	
  
});


app.post('/predicate_process', function (req, res) {
  	console.log( " Filter Predicate Processing");
	let predicate = req.body.predicate;
	let kb = req.body.kb;
	var edge_type;
	var r_embedding;
	if (kb == "dbpedia") {
		kb_basedir = "../data/dbpedia/"
	}

  	shell.exec('cd public/Benchmark/Extract_metadata && python ./triples_generators.py ' + 
  			kb_basedir + 
  			' -r ' + predicate, function(code, stdout, stderr) {
	  	// stdout = stdout.replace(/'/g, '"');
	  	// res.send({info: stdout});
		if (stderr !="") {
			res.send({info:stderr});
		}
		else {
			res.send({info:stdout});
		}

	});
  
});

app.post('/generate', function (req, res) {
  	//res.render('index');
  	//console.log(req.body.city);
  	console.log("Generating Scenario ...");

  	let kb = req.body.kb;
  	let predicate = req.body.predicate;
  	// let relate_predicate = req.body.relate_predicate;
  	// relate_predicate = relate_predicate.map(item => item.split(' -- ')[0])
  	// console.log(relate_predicate);
  	// let level = req.body.level;
  	let mode = req.body.mode;
	var size_scenario;
  	var r_consist;
  	var r_homo;
  	var r_pos_neg;
  	var popular;
  	var check = true;
  	var info = ""
  	if (!req.body.predicate) {
  		info = info +  "-- Prediate not found";
  		check = false;
  	}

  	if (check) {
	  	if (mode == "train") {
		  	size_scenario = req.body.train_size_scenario;
		  	r_consist = req.body.train_consist;
		  	r_homo = req.body.train_homo;
		  	r_pos_neg = req.body.train_pos_neg;
		  	popular = req.body.train_popular;
	    }
	    else {
		    size_scenario = req.body.test_size_scenario;
		  	r_consist = req.body.test_consist;
		  	r_homo = req.body.test_homo;
		  	r_pos_neg = req.body.test_pos_neg;
		  	popular = req.body.test_popular;
	    }
		if (kb == "dbpedia") {	 			
			kb_basedir = "../data/dbpedia/"
			    
	  	}
	  	let train_file_ifexist = kb_basedir + "splits/" + predicate + "/" + "train" + "_c" + parseFloat(req.body.train_consist).toFixed(1).toString() + "_h" + parseFloat(req.body.train_homo).toFixed(1).toString() + "_s" + req.body.train_size_scenario + "_r" + req.body.train_pos_neg + "_" + req.body.train_popular + "_scenario.tsv"
	  	shell.exec('cd public/Benchmark/Extract_metadata &&' + 'python ./scenario_generator.py ' + 
	  		kb_basedir + ' -r ' + predicate + ' -t ' + 
	  		mode +' -sc ' + size_scenario + ' -hm ' + r_homo + ' -cs ' + r_consist + ' -pn ' + r_pos_neg + ' -p ' + popular + ' -f '  + train_file_ifexist, function(code, stdout, stderr) {
	  			  	//shell.exec('python ./sc.py' + ' -r ' + predicate + ' -e ' + spec_file["graphs"]["relation sets"]["relation_file"] + ' -i ' +  spec_file["split"])	
	  			// console.log("Info...")
	  			console.log(stderr)
	  			output_filename = "/Benchmark" + kb_basedir.replace("..", "") + "splits/" + predicate + "/" + mode + "_c" + parseFloat(r_consist).toFixed(1).toString() + "_h" + parseFloat(r_homo).toFixed(1).toString() + "_s" + size_scenario + "_r" + r_pos_neg + "_" + popular +  "_scenario.tsv"
	  			if (stderr !="") {
	   				res.send({info:"Error..", output: [null]});
	   			}
	   			else { 		
	   				res.send({info:stdout, output: output_filename}); 					 
	   				}
	   			
	   			console.log("Generation completed...");
	  		});
  	}
  	else {
  		res.send({info:info, output: [null]});
  	}
})

app.post('/fact_check', function (req, res) {
	// res.setTimeout(12000000, function(){
  	//res.render('index');
  	//console.log(req.body.city);
  	console.log("Generating spec file...");

  	let algo = req.body.algorithm;
  	let kb = req.body.kb;
  	let predicate = req.body.predicate;

  	// let level = req.body.level;
  	
  	// let size_train_scenario = req.body.size_scenario; 
  	var check = true;

  	var info = "";
  	if (!req.body.predicate) {
  		info = info +  "-- Prediate not found";
  		check = false;
  	}

	try {
	    fs.statSync("./public/Benchmark/data/"+ kb + "/splits/" + predicate + "/" + "train" + "_c" + parseFloat(req.body.train_consist).toFixed(1).toString() + "_h" + parseFloat(req.body.train_homo).toFixed(1).toString() + "_s" + req.body.train_size_scenario + "_r" + req.body.train_pos_neg + "_" + req.body.train_popular + "_scenario.tsv");
	    console.log('train file exists');
	}
	catch (err) {
	  if (err.code === 'ENOENT') {
		check = false;
		info = info + "-- Train file not found";
	    console.log("./public/Benchmark/data/"+ kb + "/splits/" + predicate + "/" + "train" + "_c" + parseFloat(req.body.train_consist).toFixed(1).toString() + "_h" + parseFloat(req.body.train_homo).toFixed(1).toString() + "_s" + req.body.train_size_scenario + "_r" + req.body.train_pos_neg + "_" + req.body.train_popular + "_scenario.tsv");
	  }
	}

	try {
	    fs.statSync("./public/Benchmark/data/"+ kb + "/splits/" + predicate + "/" + "test" + "_c" + parseFloat(req.body.test_consist).toFixed(1).toString() + "_h" + parseFloat(req.body.test_homo).toFixed(1).toString() + "_s" + req.body.test_size_scenario + "_r" + req.body.test_pos_neg + "_" + req.body.test_popular + "_scenario.tsv");
	    console.log('test file exists');
	}
	catch (err) {
	  if (err.code === 'ENOENT') {
		check = false;
		info = info + "-- Test file not found";
	    console.log("./public/Benchmark/data/"+ kb + "/splits/" + predicate + "/" + "test" + "_c" + parseFloat(req.body.test_consist).toFixed(1).toString() + "_h" + parseFloat(req.body.test_homo).toFixed(1).toString() + "_s" + req.body.test_size_scenario + "_r" + req.body.test_pos_neg + "_" + req.body.test_popular + "_scenario.tsv");
	  }
	}

	console.log(info)
	if (check) {
	  	let spec_file = {};
	  	if (kb == "dbpedia") {
	  		var kb_basedir = "../data/dbpedia/"
			spec_file["graph"] =  {"name":"dbpedia", 
				// "type": "remote",
				"relation_sets":[{ 
				"relation file":"labeled_edges.tsv", 
				"is kb":true}]};
	  	}

		let train_file =  kb_basedir + "splits/" + predicate + "/" + "train" + "_c" + parseFloat(req.body.train_consist).toFixed(1).toString() + "_h" + parseFloat(req.body.train_homo).toFixed(1).toString() + "_s" + req.body.train_size_scenario + "_r" + req.body.train_pos_neg + "_" + req.body.train_popular + "_scenario.tsv";
  		let test_file =   kb_basedir + "splits/" + predicate + "/" + "test" + "_c" + parseFloat(req.body.test_consist).toFixed(1).toString() + "_h" + parseFloat(req.body.test_homo).toFixed(1).toString() + "_s" + req.body.test_size_scenario + "_r" + req.body.test_pos_neg + "_" + req.body.test_popular + "_scenario.tsv";

	  	spec_file["split"] = {"name":  predicate,
	 	 	
	 	 				  "train_file": train_file,
	 	 				  "test_file": test_file};

	 	var auroc = "0.0"

	 	if (algo == "sfe") {
	 		let steps = Number(req.body.subgraph_depth);
	 		let score_file =  kb_basedir + "splits/" + predicate + "/" + "test" + "_c" + parseFloat(req.body.test_consist).toFixed(1).toString() + "_h" + parseFloat(req.body.test_homo).toFixed(1).toString() + "_s" + req.body.test_size_scenario + "_r" + req.body.test_pos_neg + "_" + req.body.test_popular + "_sfe_score.tsv";
		    spec_file["operation"] = {
					  "features": { 
					  	"type" : "subgraphs", 
						 "path finder": {
						 	    "type": "BfsPathFinder",
	        					"number of steps": steps
						 },
						 "feature extractors": [
	        					"PraFeatureExtractor",
	      				],
	      				"feature size": -1
	    				},
	    				"learning": {
	      					"l1 weight": 0.5,
	      					"l2 weight": 0.01
	    				}
					   } 

			content = JSON.stringify(spec_file) 	
			var spec_dir = path.join(__dirname + "/public/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json")
			fs.writeFile(spec_dir, content, 'utf8', function (err) {
			    if (err) {
			        return console.log(err);
			    }
			    console.log("Experiment spec file was saved!");		
			    shell.exec("cd public/Benchmark/pra_sfe &&" + "rm -rf " + kb_basedir + "in_progress &&"  + "sbt 'run-main edu.cmu.ml.rtw.pra.experiments.ExperimentRunner   "  + kb_basedir + 
		                  " "   + algo + ".json" + "'", function(code, stdout, stderr) {
		                  		// if (stdout == "") {

								  	shell.exec('cd public/Benchmark/Extract_metadata && python compute_auc.py  ' + 
								  			" "   + score_file, function(code, stdout, stderr) {
									if (stderr == "") {
									  	stdout = stdout.replace(/'/g, '"');
									  	auroc  = stdout;
									  	console.log(stdout)
									  	//res.send(JSON.parse(stdout));	
									  	console.log("Fact checking done...");  	
		  								//shell.exec('python ./sc.py' + ' -r ' + predicate + ' -e ' + spec_file["graphs"]["relation sets"]["relation_file"] + ' -i ' +  spec_file["split"])	
		   								res.send({spec: "/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json", output:"/Benchmark" + score_file.replace("..", ""), auroc: auroc});
		   							}
		   							else {
		   								res.send({spec: "/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json", output:[null], auroc: stderr});
		   							}
									});
								 // }
								 // else {
								 // 	res.send({spec: "/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json", output:[null], auroc: "Error!!! maybe graph not loaded yet or out of memory issue..."});
								 // }
		                  });	
			}); 	
	 	}

		if (algo == "kl") {	
			let closure = req.body.closure;
			let n_procs = 6;		
			let weightype = req.body.weighttype;
			let is_directed = req.body.is_directed;
			let score_file =  kb_basedir + "splits/" + predicate + "/" + "test" + "_c" + parseFloat(req.body.test_consist).toFixed(1).toString() + "_h" + parseFloat(req.body.test_homo).toFixed(1).toString() + "_s" + req.body.test_size_scenario + "_r" + req.body.test_pos_neg + "_" + req.body.test_popular + "_klinker_score.tsv";
		    spec_file["operation"] = {"type" : "confmatrix", 
					  "features": { 
					 	"nprocs" :  n_procs , 
						"closure" :  closure , 
						"weight" :  weightype, 
						"is_directed" :  is_directed
					   } } 	
			content = JSON.stringify(spec_file)
			var spec_dir = path.join(__dirname + "/public/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json")
			fs.writeFile(spec_dir, content, 'utf8', function (err) {
			    if (err) {
			        return console.log(err);
			    }
			    console.log("Experiment spec file was saved!");
				shell.exec('cd public/Benchmark/knowledge_linker &&' + 'klinker confmatrix ' + kb_basedir + 
		                " " + algo + ".json" ,function(code, stdout, stderr) {
		                		console.log(stderr);
		                		console.log(stdout);
		                  		// if (stderr == "") {
								  	shell.exec('cd public/Benchmark/Extract_metadata && python compute_auc.py  ' + 
								  			" "   + score_file, function(code, stdout, stderr) {
									if (stderr == "") {
									  	stdout = stdout.replace(/'/g, '"');
									  	auroc  = stdout;
									  	console.log(stdout)
									  	//res.send(JSON.parse(stdout));	
									  	console.log("Fact checking done...");  	
		  								//shell.exec('python ./sc.py' + ' -r ' + predicate + ' -e ' + spec_file["graphs"]["relation sets"]["relation_file"] + ' -i ' +  spec_file["split"])	
		   								res.send({spec: "/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json", output:"/Benchmark" + score_file.replace("..", ""), auroc: auroc});
		   							}
		   							else {
		   								res.send({spec: "/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json", output:[null], auroc: stderr});
		   							}
									});
								 // }
								 // else {
								 // 	res.send({spec: "/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json", output:[null], auroc: stderr});
								 // }
		                  });	
			});
	  	}

	  	if (algo == "kgminer") {
			let max_depth = req.body.max_depth;
			let n_procs = 5;		
			let is_directed = req.body.is_directed;
			let score_file =  kb_basedir + "splits/" + predicate + "/" + "test" + "_c" + parseFloat(req.body.test_consist).toFixed(1).toString() + "_h" + parseFloat(req.body.test_homo).toFixed(1).toString() + "_s" + req.body.test_size_scenario + "_r" + req.body.test_pos_neg + "_" + req.body.test_popular + "_kgminer_score.tsv";
		    spec_file["operation"] = { 
					  "features": { 
					 	"nprocs" :  n_procs , 
						"max_depth" :  max_depth, 
						"is_directed" :  is_directed
					   } } 	
			content = JSON.stringify(spec_file) 	
			var spec_dir = path.join(__dirname + "/public/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json")
			fs.writeFile(spec_dir, content, 'utf8', function (err) {
			    if (err) {
			        return console.log(err);
			    }
			    console.log("Experiment spec file was saved!");		
			    shell.exec("cd public/Benchmark/kg_miner &&" + "python run_test.py  "  + kb_basedir + 
		                  " "   + algo + ".json" , function(code, stdout, stderr) {
		                  		// if (stderr == "") {
								  	shell.exec('cd public/Benchmark/Extract_metadata && python compute_auc.py  ' + 
								  			" "   + score_file, function(code, stdout, stderr) {
									if (stderr == "") {
									  	stdout = stdout.replace(/'/g, '"');
									  	auroc  = stdout;
									  	console.log(stdout)
									  	//res.send(JSON.parse(stdout));	
									  	console.log("Fact checking done...");  	
		  								//shell.exec('python ./sc.py' + ' -r ' + predicate + ' -e ' + spec_file["graphs"]["relation sets"]["relation_file"] + ' -i ' +  spec_file["split"])	
		   								res.send({spec: "/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json", output:"/Benchmark" + score_file.replace("..", ""), auroc: auroc});
		   							}
		   							else {
		   								res.send({spec: "/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json", output:[null], auroc: stderr});
		   							}
									});
								 // }
								 // else {
								 // 	res.send({spec: "/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json", output:[null], auroc: stderr});
								 // }
		                  });	
			});
	  	}

	  	if (algo == "rule") {
			let embed_dim = req.body.embedsize;
			let method = "Rule";
			
			let lr = req.body.learnrate;
			let score_file =  kb_basedir + "splits/" + predicate + "/" + "test" + "_c" + parseFloat(req.body.test_consist).toFixed(1).toString() + "_h" + parseFloat(req.body.test_homo).toFixed(1).toString() + "_s" + req.body.test_size_scenario + "_r" + req.body.test_pos_neg + "_" + req.body.test_popular + "_rule_score.tsv";
		    spec_file["operation"] = { 
		    		  "method": method 
		    		  } 	
			content = JSON.stringify(spec_file) 	
			var spec_dir = path.join(__dirname + "/public/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json")
			fs.writeFile(spec_dir, content, 'utf8', function (err) {
			    if (err) {
			        return console.log(err);
			    }
			    console.log("Experiment spec file was saved!");		

			    shell.exec("cd public/Benchmark/rule_based &&  " + "python run.py "  + kb_basedir + 
		                  " "   + algo + ".json", function(code, stdout, stderr) {
		                  		// if (stderr == "") {
								  	shell.exec('cd public/Benchmark/Extract_metadata && python compute_auc.py  ' + 
								  			" "   + score_file, function(code, stdout, stderr) {
									if (stderr == "") {
									  	stdout = stdout.replace(/'/g, '"');
									  	auroc  = stdout;
									  	console.log(stdout)
									  	//res.send(JSON.parse(stdout));	
									  	console.log("Fact checking done...");  	
		  								//shell.exec('python ./sc.py' + ' -r ' + predicate + ' -e ' + spec_file["graphs"]["relation sets"]["relation_file"] + ' -i ' +  spec_file["split"])	
		   								res.send({spec: "/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json", output:"/Benchmark" + score_file.replace("..", ""), auroc: auroc});
		   							}
		   							else {
		   								res.send({spec: "/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json", output:[null], auroc: stderr});
		   							}
									});
								 // }
								 // else {
								 // 	res.send({spec: "/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json", output:[null], auroc: stderr});
								 // }
		                  });	
			});
	  	}


	  	if (algo == "transe") {
			let embed_dim = req.body.embedsize;
			let method = "TransE";
			let n_procs = 5;	
			let nepoches = 1500;	
			let lr = req.body.learnrate;
			let score_file =  kb_basedir + "splits/" + predicate + "/" + "test" + "_c" + parseFloat(req.body.test_consist).toFixed(1).toString() + "_h" + parseFloat(req.body.test_homo).toFixed(1).toString() + "_s" + req.body.test_size_scenario + "_r" + req.body.test_pos_neg + "_" + req.body.test_popular + "_transe_score.tsv";
		    spec_file["operation"] = { 
		    		  "method": method,
					  "features": { 
					      "embed_dim": embed_dim,
					      "nprocs": n_procs,
					      "learning_rate": lr,
					      "nepoches": nepoches
					   } } 	
			content = JSON.stringify(spec_file) 	
			var spec_dir = path.join(__dirname + "/public/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json")
			fs.writeFile(spec_dir, content, 'utf8', function (err) {
			    if (err) {
			        return console.log(err);
			    }
			    console.log("Experiment spec file was saved!");		

			    shell.exec("cd public/Benchmark/para_graphE && " + "python run_test.py "  + kb_basedir + 
		                  " "   + algo + ".json", function(code, stdout, stderr) {
		                  		// if (stderr == "") {
								  	shell.exec('cd public/Benchmark/Extract_metadata && python compute_auc.py  ' + 
								  			" "   + score_file, function(code, stdout, stderr) {
									if (stderr == "") {
									  	stdout = stdout.replace(/'/g, '"');
									  	auroc  = stdout;
									  	console.log(stdout)
									  	//res.send(JSON.parse(stdout));	
									  	console.log("Fact checking done...");  	
		  								//shell.exec('python ./sc.py' + ' -r ' + predicate + ' -e ' + spec_file["graphs"]["relation sets"]["relation_file"] + ' -i ' +  spec_file["split"])	
		   								res.send({spec: "/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json", output:"/Benchmark" + score_file.replace("..", ""), auroc: auroc});
		   							}
		   							else {
		   								res.send({spec: "/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json", output:[null], auroc: stderr});
		   							}
									});
								 // }
								 // else {
								 // 	res.send({spec: "/Benchmark" + kb_basedir.replace("..", "") + "experiment_specs/" + algo +".json", output:[null], auroc: stderr});
								 // }
		                  });	
			});
	  	}

  	
	}
	else {
		res.send({spec: [null], output:[null], auroc: info});
	}

	  	//console.log('python ./public/Benchmark_Fact_Checking/Extract_metadata/sc.py' + ' -r ' + predicate + ' -e ' + spec_file["graphs"]["relation_file"] + ' -i ' +  spec_file["split"]);
	})

// })

var server = app.listen(5006, function () {
  console.log('Example app listening on port 5006!')
})
server.timeout = 1000*60*1000;
