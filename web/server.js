const express = require('express');
const bodyParser = require('body-parser');
const app = express();
const path = require('path');
const { spawn } = require('child_process');
const uuid = require('uuidv4')


var requests = {};
var child = null;
var stdout = "";

app.set('port', process.env.PORT || 3000);
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

// TODO: attach a uuid to the body, so that the shim can echo it back to us and we can handle multiple submissions at
// once
app.post('/submit_code', function(req, res) {
    if (child === null) {
    	setup_child();
    }
    req.body.uuid = uuid();
    requests[req.body.uuid] = res;
    child.stdin.write(JSON.stringify(req.body) + '\n');
});

// TODO: this doesn't error if port in use?
app.listen(app.get('port'), function(){
  console.log('listening on *:' + app.get('port'));
});//.on('error', function(err) { if (err.errno === 'EADDRINUSE') { console.log('port busy'); } else { console.log(err); } });

function handle_stdout(data) {
    try {
	data = JSON.parse(data.toString());
    } catch (e) {
	return;
    }
    var res = requests[data.uuid];
    delete(requests[data.uuid]);
    delete(data.uuid);
    res.json(data);
    res.end();
}


function setup_child() {
    child = spawn('python', ['../scripts/shim.py'], {stdio: 'pipe'});

    //child.stdout.setEncoding('utf8')
    // TODO: make sure the JSON message is complete?
    child.stdout.on('data', function(data) {
    	stdout += data.toString();
    	stdout = stdout.split('\n\n');
    	while (stdout.length > 1) {
    	    handle_stdout(stdout.shift());
	}
	stdout = stdout[0];
    });

    // TODO: handle this
    child.stderr.on('data', function(data) {
        console.log(data.toString())
    });

    // TODO: do we always output stderr before stdout?
    child.on('close', function(code, signal) {
        console.log('closing code: ' + code + ' ' + signal)
        child = null;
    });

    child.on('exit', function(code, signal) {
        console.log('exiting: ' + code + ' ' + signal)
    });
}

setup_child();
