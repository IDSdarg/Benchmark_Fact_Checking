TRANS_FILE = transbase.hpp transe.hpp 
COMPILE_ARG = -std=c++11 -lpthread -D_GLIBCXX_USE_NANOSLEEP -O3 -march=native

all: train 
train: $(TRANS_FILE) math_func.hpp  train.cpp
	g++ $(TRANS_FILE) math_func.hpp train.cpp -o train $(COMPILE_ARG) 
