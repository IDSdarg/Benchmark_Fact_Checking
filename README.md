# BUCKLE-Benchmark Fact-Checking
This benchmark continues our previous work **[P. Huynh, P. Papotti, 2017](http://www.eurecom.fr/fr/publication/5468/download/data-publi-5468.pdf)** on fact checking by providing the first comprehensive and publicly available infrastructure for evaluating fact checking methods across a wide range of assumption about the facts and the reference information.

It is an addition material of the publication "A Benchmark for Fact Checking Algorithms Built on Knowledge Bases", from **[P. Huynh, P. Papotti](http://www.eurecom.fr/en/publication/5996/download/data-publi-5996.pdf)**, accepted to International Conference on Information and Knowledge Management (CIKM), 2019

## Download & Install
1. The benchmark server is written by **[Nodejs](https://nodejs.org/en/download/)**. Follow `package.json` to install dependencies.
2. Follow base folders for downloading and installing benchmarking systems: `Knowledge Linker (KL) [1]`, `Discriminative Predicate Path (KGMiner) [2]`, `Subgraph Feature Extraction (SFE) [3]`, `Parallel Graph Embedding (Para_GraphE) [4]`, `Rule Discover [5]` 
   
   Each system has some modifications from the original version, according to the experiments considered in this work.
   
## References:
1. https://github.com/glciampaglia/knowledge_linker
2. https://github.com/nddsg/KGMiner
3. https://github.com/matt-gardner/pra
4. https://github.com/LIBBLE/LIBBLE-MultiThread/tree/master/ParaGraphE
5. https://github.com/stefano-ortona/rudik







