var express = require('express');
var app = express();
var http = require('http').Server(app);
var io = require('socket.io')(http);
var path = require('path');
var fs = require('fs');
var execSync = require('child_process').execSync;
var uuid = require('uuid/v4');


app.use(express.static(path.join(__dirname, 'public')));

io.on('connection', function(socket){
  console.log('a user connected');
});

http.listen(3000, function(){
  console.log('listening on *:3000');
});

var max_timeout = 10;
var timeout_interval = 0.01;
var files = {};

function parse_results(file) {
    var results_file = '/mnt/e/server_tasks/.' + file + '-results'
    if (!fs.existsSync(results_file)) {
    	files[file].retries++;
        if (files[file].retries * timeout_interval > max_timeout) {
            try {
		files[file].socket.emit('response', { success:false, results: 'Server took too long to respond!' });
		fs.unlinkSync('/mnt/e/server_tasks/' + file);
		fs.unlinkSync(results_file);
	    } catch(e) {console.log(e)}
            delete(files[file])
            return;
        }
	return setTimeout(function() { parse_results(file) }, timeout_interval);
    }
    try {
    	results = fs.readFileSync(results_file);
	json = JSON.parse(results);
	files[file].socket.emit('response', { success:true, results: json });
    } catch(e) {
    	console.log(e);
    	files[file].socket.emit('response', { success:false, results: e });
    }
    delete(files[file])
    // XXX should probably clean all these up on server boot
    fs.unlinkSync(results_file);
}

io.on('connection', function(socket){
  socket.on('code', function(msg){
    file = uuid() + '.c';
    files[file] = {'socket': socket, 'retries': 0}
    fs.writeFileSync('/mnt/e/server_tasks/' + file, msg);
    setTimeout(function() { parse_results(file) }, timeout_interval);
  });
});
