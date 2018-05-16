#!/bin/bash

for i in caesar; do # credit greedy mario recover resize dictionary vigenere; do
    cd $i
    for k in *; do
	if [[ -f $k ]]; then continue; fi
	if [ ! -f "$k/output.json" ]; then continue; fi
	cd ..
	echo $i, $k
	if (( 0x${j} % 50 == 0 )); then
	    children=$(($(pgrep --parent $$ | wc -l) - 1))
	    while (( children > 50 )); do
		sleep 1
		children=$(($(pgrep --parent $$ | wc -l) - 1))
	    done
	fi
	# for dictionary, current doesn't handle other C files
	python3 move.py $i $k &
	cd $i
    done
    cd ..
done
