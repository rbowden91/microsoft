export TF_CPP_MIN_LOG_LEVEL=2

preprocess:
	#repair50_preprocess vigenere.c cs50_data/s3_graded_psets/vigenere/correct/vigenere_1.json/ cs50_data/training_data/vigenere1000 --num_processes=32 --num_files=1000 --unit_tests=cs50_data/unit_tests/vigenere/vigenere_1.json
	cgexec --sticky -g memory:repair50 repair50_preprocess vigenere.c cs50_data/s3_graded_psets/vigenere/correct/vigenere_1.json/ cs50_data/training_data/vigenere1000 --num_processes=32 --num_files=1000 --unit_tests=cs50_data/unit_tests/vigenere/vigenere_1.json --force
train:
	repair50_train -b1 -d"d2" -e3 -p cs50_data/training_data/vigenere1000 -s tmp --num_layers=2 --hidden_size=50 -t'argc==4' --num_processes=1 --cpu -f

server:
	repair50_shim -t'argc==4' -d cs50_data/training_data/tmp/tmp -p 12347

server1:
	repair50_shim -t'a,a,a|A,!,!|baz,"World, Say Hello!","Xoqmd, Rby Gflkp!"' -d cs50_data/training_data/vigenere1000 -stmp -p 12347
train1:
	repair50_train -b1 -d"d2" -e3 -p cs50_data/training_data/vigenere1000 -s tmp --num_layers=2 --cpu -t'a,a,a|A,!,!|baz,"World, Say Hello!","Xoqmd, Rby Gflkp!"' --hidden_size=50 --num_processes=16 --force

server2:
	#repair50_shim -t"argc==1|argc==4|key isn't alphabetical|A,A,A" -d cs50_data/training_data/vigenere1000/tmp -p 12347
	cgexec --sticky -g memory:repair50 repair50_shim -t"argc==1|argc==4|key isn't alphabetical|A,A,A|baz,barfoo,caqgon" -d cs50_data/training_data/vigenere1000 -stmp -p 12347

train2:
	cgexec --sticky -g memory:repair50 repair50_train -b1 -d"d2" -e3 -p cs50_data/training_data/vigenere1000 -s tmp --num_layers=2 -t"argc==1|argc==4|key isn't alphabetical|A,A,A|baz,barfoo,caqgon" --hidden_size=50 --num_processes=16 --cpu -f
	#cgexec --sticky -g memory:repair50 repair50_train -b1 -d"d2" -e2 -p cs50_data/training_data/vigenere1000 -s tmp --num_layers=2 --cpu -t'null|baz,worl!d,xoqm!d' --hidden_size=50 --num_processes=32 -f

server3:
	repair50_shim -t'null|baz,worl!d,xoqm!d|BAZ,BARFOO,CAQGON|BaZ,BaRFoo,CaQGon' -d cs50_data/training_data/vigenere1000 -stmp -p 12347
train3:
	repair50_train.exe -b1 -d"d2" -e3 -p "cs50_data/training_data/vigenere1000" -s tmp --num_layers=2 -t"null|BAZ,BARFOO,CAQGON|baz,worl!d,xoqm!d|BaZ,BaRFoo,CaQGon" --hidden_size=50 --num_processes=16 --force

multi_server:
	repair50_multi_shim -p 12344

.PHONY: server preprocess train
