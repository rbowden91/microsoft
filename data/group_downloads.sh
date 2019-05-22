#!/bin/bash

# XXX fork invalid argument?
cd cs50_data/s3_unsorted_downloads
for k in initials caesar vigenere greedy credit mario recover resize dictionary fifteen; do
    mkdir -p ../s3_sorted_psets/$k
    find . -type f -name "$k".c | perl -pe 's|(\./.*?/.*?/).*|\1|' | xargs -n 1000 cp -r -t ../s3_sorted_psets/$k/
done
