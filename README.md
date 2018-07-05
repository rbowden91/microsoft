* data/

A directory used to save preprocessed vigenere.c files and trained models.

* fake_libc_include/

Used by files that need to use pycparser to preprocess and parse C files. Taken straight from the pycparser repo, but
with cs50.h added.

* linear/

Token-based approach.

* search/

Enumerative search (may become a subdirectory of linear/, and a new enumerative search directory will go under tree/)

* tasks/

A directory for web/server.js (the nodejs server for the web interface where students can submit code) and
{linear,tree}/server.js (the Python servers that takes in student code and writes back the heatmap files) to write files
for communication. This allows for communication between them, regardless of whether one is running under the Command
Prompt and the other is running in the Bash subshell.

* tree/

Tree-based approach.

* vigenere/

A subset of the check50 submissions for vigenere.c. The pset spec changed slightly between 2015 and 2016, so within
correct15/ are solutions that pass the 2015 checks, and within correct16/ are solutions that pass the 2016 checks.
invalid/ contains files that are invalid for some reason (for example, pycparser couldn't parse them), no_compile/
contains files that did not compile, and incorrect/ contains the remaining submissions that at least compile but pass
neither the 2015 nor the 2016 checks.

* web/

The frontend for student code submission and heatmap display.








1)
./parse.sh
    Runs dump_ast.py on vigenere.c files in ../vigenere/correct15, generating tree_stripped.json for each file

2)
python3 tree_read.py ../vigenere/correct15/ ../data/tree5
    Generates data files (where tokens have been converted to ids) from the tree_stripped.json files.
    The "num_files" argument at the top of the file can be used to control how many files in the input_path should be
    used.
