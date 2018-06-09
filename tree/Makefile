server:
	python3 server.py --data_path=data/vig_no_decl100/ast/tmp

preprocess:
	python3 preprocess.py vigenere.c ../problems/vigenere_old/correct15 data/vig_no_decl100 --num_files=100 -t=16

train:
	python3 train.py -n100 -b1 -mast -d"d3" -e1 -f -p./data/vig_no_decl10

.PHONY: server preprocess train
