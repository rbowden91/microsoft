cd ../vigenere/correct15
count=0
for i in 1*; do
    echo $i
    cd $i
    path=`pwd`
    cd ../../../tree
    python3 ../preprocces.py "$path/vigenere.c" > "$path/normalized_vigenere.c"
    python3 dump_ast.py "$path/vigenere.c" > "$path/tree_stripped.json"
    cd ../vigenere/correct15
    (( count++ ))
    echo $count
done
