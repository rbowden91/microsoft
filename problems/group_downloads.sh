#!/bin/bash

# XXX fork invalid argument?
cd tmp
for k in vigenere greedy credit mario recover resize dictionary; do
    mkdir -p ../psets/$k
    find . -type f -name "$k".c | perl -pe 's|(\./.*?/.*?/).*|\1|' | xargs -n 1000 mv -t ../psets/$k/
done
