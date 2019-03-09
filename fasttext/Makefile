#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

CXX = c++
CXXFLAGS = -pthread -std=c++0x
OBJS = args.o dictionary.o matrix.o vector.o model.o utils.o fasttext.o
INCLUDES = -I.

opt: CXXFLAGS += -O3 -funroll-loops
opt: fasttext

debug: CXXFLAGS += -g -O0 -fno-inline
debug: fasttext

args.o: args.cc args.h
	$(CXX) $(CXXFLAGS) -c args.cc

dictionary.o: dictionary.cc dictionary.h args.h
	$(CXX) $(CXXFLAGS) -c dictionary.cc

matrix.o: matrix.cc matrix.h utils.h
	$(CXX) $(CXXFLAGS) -c matrix.cc

vector.o: vector.cc vector.h utils.h
	$(CXX) $(CXXFLAGS) -c vector.cc

model.o: model.cc model.h args.h
	$(CXX) $(CXXFLAGS) -c model.cc

utils.o: utils.cc utils.h
	$(CXX) $(CXXFLAGS) -c utils.cc

fasttext.o: fasttext.cc *.h
	$(CXX) $(CXXFLAGS) -c fasttext.cc

fasttext: $(OBJS) fasttext.cc
	$(CXX) $(CXXFLAGS) $(OBJS) main.cc -o fasttext

clean:
	rm -rf *.o fasttext
