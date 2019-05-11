SHELL := /bin/bash

export TF_CPP_MIN_LOG_LEVEL=2

preprocess:
	cgexec --sticky -g memory:robo50 robo50_preprocess vigenere.c cs50_data/s3_graded_psets/vigenere/correct/vigenere_1.json/ cs50_data/training_data/vigenere1000_3 --num_processes=32 --num_files=1000 --unit_tests=cs50_data/unit_tests/vigenere/vigenere_1.json --force

server:
	cgexec --sticky -g memory:robo50 robo50_shim -t'argc==1|argc==4|key isn'"'"'t alphabetical|A,A,A|B,C,D|b,c,d|a,a,a|A,!,!|baz,"World, Say Hello!","Xoqmd, Rby Gflkp!"|baz,barfoo,caqgon|null|BAZ,BARFOO,CAQGON|baz,worl!d,xoqm!d|BaZ,BaRFoo,CaQGon' -d cs50_data/training_data/vigenere1000_3 -stmp -p 12347 --servers="korra.rbowden.com:12347|aang.rbowden.com:12347"

########################################################3

train1:
	#cgexec --sticky -g memory:robo50 robo50_train -b1 -d"d2" -e3 -p cs50_data/training_data/vigenere1000_2 -s tmp --num_layers=2 -t"argc==1|argc==4|key isn't alphabetical|A,A,A|B,C,D|b,c,d" --hidden_size=50 --num_processes=16 --cpu -f
	cgexec --sticky -g memory:robo50 robo50_train -b1 -d"d2" -e3 -p cs50_data/training_data/vigenere1000_3 -s tmp --num_layers=2 -t"key isn't alphabetical|A,A,A|B,C,D|b,c,d" --hidden_size=50 --num_processes=16 --cpu -f

server1:
	if [ ! -d "/sys/fs/cgroup/memory/robo50" ]; then
	    sudo cgcreate -t rbowden:rbowden -a rbowden:rbowden -g memory:robo50
	    echo $$(( 20 * 1024 * 1024 * 1024 )) > /sys/fs/cgroup/memory/robo50/memory.limit_in_bytes
	    echo $$(( 20 * 1024 * 1024 * 1024 )) > /sys/fs/cgroup/memory/robo50/memory.memsw.limit_in_bytes
	fi
	#cgexec --sticky -g memory:robo50 robo50_shim -t"key isn't alphabetical|A,A,A|B,C,D|b,c,d" -d cs50_data/training_data/vigenere1000_3 -stmp -p 12347 --servers="korra.rbowden.com:12347|aang.rbowden.com:12347"
	cgexec --sticky -g memory:robo50 robo50_shim -t"argc==1|argc==4|key isn't alphabetical|A,A,A|B,C,D|b,c,d" -d cs50_data/training_data/vigenere1000_3 -stmp -p 12347 --servers="korra.rbowden.com:12347|aang.rbowden.com:12347"

########################################################3

train2:
	robo50_train -b1 -d"d2" -e3 -p cs50_data/training_data/vigenere1000_3 -s tmp --num_layers=2 --cpu -t'a,a,a|A,!,!|baz,"World, Say Hello!","Xoqmd, Rby Gflkp!"|baz,barfoo,caqgon' --hidden_size=50 --num_processes=16 --force

server2:
	robo50_shim -t'a,a,a|A,!,!|baz,"World, Say Hello!","Xoqmd, Rby Gflkp!"|baz,barfoo,caqgon' -d cs50_data/training_data/vigenere1000_3 -stmp -p 12347

########################################################3

train3:
	robo50_train -b1 -d"d2" -e3 -p "cs50_data/training_data/vigenere1000_3" -s tmp --num_layers=2 -t"null|BAZ,BARFOO,CAQGON|baz,worl!d,xoqm!d|BaZ,BaRFoo,CaQGon" --hidden_size=50 --num_processes=16 --force --cpu

server3:
	robo50_shim -t'null|baz,worl!d,xoqm!d|BAZ,BARFOO,CAQGON|BaZ,BaRFoo,CaQGon' -d cs50_data/training_data/vigenere1000_3 -stmp -p 12347

.PHONY: preprocess train1 server1 train2 server2 train3 server3
.ONESHELL:
