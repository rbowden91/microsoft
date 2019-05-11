#!/usr/bin/env bash
cd ~/repos/robo50/
git commit -am "update" && git push

server_num=1
for server in korra_local appa_local aang_bash; do
    cmd="tmux send-keys -t robo_server.0 'cd ~/repos/robo50/robo50/data && make server$server_num' ENTER"
    ssh -t $server "cd ~/repos/robo50 && git pull && git submodule update && $cmd"
    ((server_num++))
done

cd robo50/data
make multi_server
