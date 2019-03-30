tmux send-keys -t repair_server.0 "" ENTER

ssh -t "cd ~/repos/repair50/

cd ~/repos/repair50/centipyde
git pull
cd ~/repos/repair50/data
git pull
make server1
