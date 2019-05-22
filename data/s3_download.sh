#!/bin/bash

download () {
    local num="$1"
    mkdir -p $num
    cd $num
    #https://github.com/andrewrk/node-s3-cli
    #sudo npm install -g s3-cli
    while IFS= read -r line; do
	line=$(echo "$line" | cut -b '13-')
	echo "$line"
	s3-cli sync s3://sandbox50x/uploads/$line "$line" 2> /dev/null
    done < <(s3-cli ls s3://sandbox50x/uploads/$num)
}

mkdir -p cs50_data/s3_unsorted_downloads
cd cs50_data/s3_unsorted_downloads
for i in $(seq -f "%g" 50000 65535); do
    num=`printf "%04x" $i`
    children=$(($(pgrep --parent $$ | wc -l) - 1))
    while (( children > 100 )); do
    	sleep 1
	children=$(($(pgrep --parent $$ | wc -l) - 1))
    done
    download "$num" &
done
