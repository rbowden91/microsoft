#!/bin/bash
declare -A arr=()

for i in greedy; do #mario recover resize dictionary vigenere; do
    cd $i
    pushd .
    for k in *; do
    	popd
    	pushd .
	cd "$k"
	if [[ ! -f "$i.c" ]]; then
	    cd ..
	    #rm -rf $k
	    continue
	fi
	echo $i, $k
	read cksm _ < <(md5sum "$i.c")
	if ((arr[$cksm]++)); then
	    # we found a duplicate submission. no need to parse
	    echo 'duplicate'
	    #rm -rf $k
	else
	    if [[ -f output.json ]]; then
		continue
	    fi
	    children=$(($(pgrep --parent $$ | wc -l) - 1))
	    while (( children > 5 )); do
		sleep 3
		children=$(($(pgrep --parent $$ | wc -l) - 1))
	    done
	    # for dictionary, current doesn't handle other C files
	    python3 ../../parse.py $i.c $i &
	fi
    done
    cd ..
done

#tr -s '\n' < caesar.c
# astyle --ascii --add-braces --break-one-line-headers --align-pointer=name --pad-comma --unpad-paren --pad-header --pad-oper --max-code-length=132 --convert-tabs --indent=spaces=4 --indent-continuation=1 --indent-switches --lineend=linux --min-conditional-indent=1 --options=none --style=allman caesar.c
# gcc -fpreprocessed -dD -E -P caesar.c
