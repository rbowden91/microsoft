preprocess:
	repair50_preprocess vigenere.c cs50_data/s3_graded_psets/vigenere/correct/vigenere_1.json/ cs50_data/training_data/vigenere1000 --num_files=1000 --unit_tests=cs50_data/unit_tests/vigenere/vigenere_1.json

train:
	repair50_train -n1000 -b1 -mast -d"d2" -e6 -f -p cs50_data/training_data/vigenere1000 -s tmp1

server:
	repair50_shim -p cs50_data/training_data/vig_no_decl10/ast/tmp

.PHONY: server preprocess train
