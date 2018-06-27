const express = require('express');
const bodyParser = require('body-parser');
const app = express();
const path = require('path');
const { spawn } = require('child_process');


var child = null;

app.set('port', process.env.PORT || 3000);

app.use(bodyParser.urlencoded({ extended: true }));

app.use(express.static(path.join(__dirname, 'public')));
app.post('/submit_code', function(req, res) {
    console.log(req.body);
    res.end("yes");
});

// TODO: this doesn't error if port in use?
app.listen(app.get('port'), function(){
  console.log('listening on *:' + app.get('port'));
});//.on('error', function(err) { if (err.errno === 'EADDRINUSE') { console.log('port busy'); } else { console.log(err); } });


function setup_child() {
    child = spawn('python', ['shim.py'], {stdio: 'pipe'});

    //child.stdout.setEncoding('utf8')
    child.stdout.on('data', function(data) {
        console.log(data.toString())
    })

    // TODO: handle this
    child.stderr.on('data', function(data) {
        console.log(data.toString())
    })

    // TODO: do we always output stderr before stdout?
    child.on('close', function(code, signal) {
        console.log('closing code: ' + code + ' ' + signal)
    })
    child.on('exit', function(code, signal) {
        console.log('exiting: ' + code + ' ' + signal)
    })
}

//setup_child(socket, msg);
