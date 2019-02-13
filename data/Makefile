preprocess:
	repair50_preprocess vigenere.c cs50_data/s3_graded_psets/vigenere/correct/vigenere_1.json/ cs50_data/training_data/vigenere1000_ptr --num_files=1000 --unit_tests=cs50_data/unit_tests/vigenere/vigenere_1.json

train:
	export TF_CPP_MIN_LOG_LEVEL=2
	repair50_train -n1000 -b1 -mast -d"d1" -e6 -f -p cs50_data/training_data/vigenere1000_ptr -s tmp3

server:
	repair50_shim -p cs50_data/training_data/vigenere1000/ast/tmp1

.PHONY: server preprocess train
