export TF_CPP_MIN_LOG_LEVEL=2

preprocess:
	repair50_preprocess vigenere.c cs50_data/s3_graded_psets/vigenere/correct/vigenere_1.json/ cs50_data/training_data/vigenere1000_tests --num_files=100 --unit_tests=cs50_data/unit_tests/vigenere/vigenere_1.json

train:
	repair50_train -n800 -b1 -d"d2" -e3 -f -p cs50_data/training_data/vigenere1000_tests -s tmp14 --num_layers=2 --cpu
	#--hidden_size=600

server:
	repair50_shim -p cs50_data/training_data/vigenere1000_tests/tmp14

.PHONY: server preprocess train
