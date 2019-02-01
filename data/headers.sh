#!/bin/bash

# find all valid headers for the problems
for i in credit recover resize vigenere dictionary; do # caesar credit greedy mario recover resize dictionary vigenere; do
    echo $i
    cd $i
    OUT=$(find . \( -type f -name '*.c' -or -name '*.h' \) -print0 | xargs -0 -n 1000 perl -nle 'print $1 if /\s*#\s*include\s*<\s*(.+?)\s*>/' | sort | uniq -c | awk '{gsub(/["\\]/,"\\\\&",$2); printf("\"%s\":%s,\n",$2,$1)}')
    echo "{${OUT%?}}" > headers.json
    cd ..
done
