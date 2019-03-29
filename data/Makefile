export TF_CPP_MIN_LOG_LEVEL=2

preprocess:
	repair50_preprocess vigenere.c cs50_data/s3_graded_psets/vigenere/correct/vigenere_1.json/ cs50_data/training_data/vigenere1000 -t16 --num_files=1000 --unit_tests=cs50_data/unit_tests/vigenere/vigenere_1.json

train:
	repair50_train -b1 -d"d2" -e2 -f -p cs50_data/training_data/vigenere1000 -s tmp15 --num_layers=2 --cpu -t'test9' --hidden_size=50

server:
	repair50_shim -t'test7 test8' -p cs50_data/training_data/vigenere1000/tmp15

.PHONY: server preprocess train
