#!/bin/bash

download () {
    local num="$1"
    mkdir -p $num
    cd $num
    s3cmd get --recursive --force s3://sandbox50x/uploads/$num
}

mkdir -p tmp
cd tmp
for i in $(seq -f "%g" 00000 65535); do
    num=`printf "%04x" $i`
    children=$(($(pgrep --parent $$ | wc -l) - 1))
    while (( children > 1000 )); do
    	sleep 1
	children=$(($(pgrep --parent $$ | wc -l) - 1))
    done
    download "$num" &
done
