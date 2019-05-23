#/usr/bin/env bash

cd cs50_data/s3_sorted_psets
rootdir=$(pwd)

if ! which fdupes ; then
    sudo apt install -y fdupes
fi
if ! which clang-format ; then
    sudo apt install -y clang-format
fi
if ! [ -f /tmp/scc/scc ]; then
    git clone https://github.com/jleffler/scc-snapshots /tmp/scc
    cd /tmp/scc
    make
fi
cd "$rootdir"
#for i in *; do
#    cd "$rootdir/$i"
#    echo $i
#    find . -type f -not -name "$i.c" -delete
#done
#cd ..

function process () {
    file="$1"
    echo "$file"
    base=$(basename "$file")
    dir=$(dirname "$file")
    cd "$rootdir/$dir"
    tmp="$(cat "$base")"
    /tmp/scc/scc <<< "$tmp" | clang-format > "$base"
}

find . -name '*.c' | while IFS= read -r file; do
    process "$file" &
    [ $( jobs | wc -l ) -ge $( nproc ) ] && wait
done

while [ $( jobs | wc -l ) -ge $( 2 ) ]; do
    wait
done

echo "fdupes running\n"
cd "$rootdir"
fdupes -rdN .
