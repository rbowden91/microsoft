cd ../vigenere/correct15
for i in *; do
    cd $i
    path=`pwd`
    cd ../../../tree
    python3 dump_ast.py "$path/vigenere.c" > "$path/tree_stripped.json"
    cd ../vigenere/correct15
done
