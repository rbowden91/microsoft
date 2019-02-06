import sys
import os
import csv

"""
echo "select username from submissions,users where branch like 'cs50/2018/fall/mario/less' and user_id=users.id group by username;" | mysql -u$MYSQL_USERNAME -h$MYSQL_HOST -p$MYSQL_PASSWORD $MYSQL_DATABASE > mario_less.tsv
"""

with open('cs50_data/submissions_with_grades.tsv') as tsv:
    next(tsv)
    for line in csv.reader(tsv, dialect="excel-tab"):
        path = os.path.join(['cs50_data', 'tf_graded_psets', line[1])
        os.makedirs(path, '2', mode=0o711, exist_ok=True)
        os.makedirs(path, '3', mode=0o711, exist_ok=True)
        os.makedirs(path, '4', mode=0o711, exist_ok=True)
        path = os.path.join(path, line[2], line[0])
        try:
            os.system('git clone --depth=1 --branch ' + line[1] + ' git@github.com:submit50/' + line[0] + ' ' + path)
            os.system('rm -rf ' + os.path.join(path, '.git'))
        except:
            pass
