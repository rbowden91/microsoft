export TF_CPP_MIN_LOG_LEVEL=2

preprocess:
	repair50_preprocess vigenere.c cs50_data/s3_graded_psets/vigenere/correct/vigenere_1.json/ cs50_data/training_data/vigenere1000 --num_processes=32 --num_files=1000 --unit_tests=cs50_data/unit_tests/vigenere/vigenere_1.json
train:
	repair50_train -b1 -d"d2" -e2 -p cs50_data/training_data/vigenere1000 -s tmp --num_layers=2 --hidden_size=50 -t'argc==4' --num_processes=16 --cpu

server:
	repair50_shim -t'null test1 test2 test3 test4 test5 test6 test7 test8 test9' -d cs50_data/training_data/vigenere1000/tmp -p 12347

server1:
	repair50_shim -t'null|baz,worl!d,xoqm!d|a,a,a|A,!,!' -d cs50_data/training_data/vigenere1000/tmp -p 12347
train1:
	repair50_train -b1 -d"d2" -e2 -p cs50_data/training_data/vigenere1000 -s tmp --num_layers=2 --cpu -t'null|baz,worl!d,xoqm!d|a,a,a|A,!,!' --hidden_size=50 --num_processes=16

server2:
	repair50_shim -t'argc==1|argc==4|key isn\'t alphabetical|A,A,A' -d cs50_data/training_data/vigenere1000/tmp -p 12347
train2:
	repair50_train -b1 -d"d2" -e2 -p cs50_data/training_data/vigenere1000 -s tmp --num_layers=2 -t'argc==1|argc==4|key isn\'t alphabetical|A,A,A' --hidden_size=50 --num_processes=16

server3:
	repair50_shim -t'baz,"World, Say Hello!","Xoqmd, Rby Gflkp!"|BAZ,BARFOO,CAQGON|BaZ,BaRFoo,CaQGon|baz,barfoo,caqgon' -d cs50_data/training_data/vigenere1000/tmp -p 12347
train3:
	repair50_train -b1 -d"d2" -e2 -p cs50_data/training_data/vigenere1000 -s tmp --num_layers=2 -t'baz,"World, Say Hello!","Xoqmd, Rby Gflkp!"|BAZ,BARFOO,CAQGON|BaZ,BaRFoo,CaQGon|baz,barfoo,caqgon' --hidden_size=50 --num_processes=16

multi_server:
	repair50_multi_shim -p 12344

.PHONY: server preprocess train
