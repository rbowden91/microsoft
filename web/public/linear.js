var color1 = [255, 0, 0];
var color2 = [0, 0, 255];
var steps = 16;

var grad = generate_gradient(color1, color2, steps);

function alphanumeric(inputtxt) {
    var letterNumber = /^[0-9a-zA-Z]+$/;
    return inputtxt.match(letterNumber);
}

// red->white->blue
function generate_gradient(color1, color2, steps) {
    var diff = [(color2[0] - color1[0]) / (steps - 1),
	    (color2[1] - color1[1]) / (steps - 1),
	    (color2[2] - color1[2]) / (steps - 1)];

    var grad = [];
    for (var i = 0; i < steps; i++) {
    	grad.push([color1[0] + i * diff[0],
		     color1[1] + i * diff[1],
		     color1[2] + i * diff[2]]);
    }
    grad.push(color2);
    return grad;
}

function gradient(grad, level) {
    var idx = Math.round((grad.length - 1)* level);
    return 'rgba(' + grad[idx].join(',') + ', 0.5)';
}

function print_indent(level) {
    var output = '';
    for (var j = 0; j < level; j++) {
	for (var k = 0; k < 4; k++) {
	    output += " ";
	}
    }
    return output;
}

function drawLinear(data) {
    // XXX for now, very rough guidelines for spacing and indentation. doesn't handle single-line ifs, loops, etc., but
    // those might not exist from pycparser?
    var output = '';
    var indentation_level = 0;
    var newline = false;
    directions = Object.keys(data[0])
    if (directions.length == 2) {
        output += '<input type="radio" name="dir" value="forward"> Forward' +
                  '<input type="radio" name="dir" value="reverse"> Reverse<br><br>'
        if (typeof(data.direction) === 'undefined') {
            data.direction = 'forward';
        }
        var direction = data.direction;
    } else {
        var direction = directions[0];
        data.direction = direction
    }
    for (var i = 0; i < data.length; i++) {
    	var token = data[i][direction].token;

	if (token === '}') {
    	    indentation_level -= 1;
	}

	if (newline) {
	    output += print_indent(indentation_level);
	    newline = false;
	}

	//if (i !== 0) {
	    //if (data[i][0] === "'")
	    output += "<span style='background-color:" + gradient(grad, data[i][direction].label_ratio) + "' data-expected='" +
		      htmlEncode(data[i][direction].label_expected) +
		      "' data-ratio='" + data[i][direction].label_ratio + "'" +
		      " data-max-prob='" + data[i][direction].label_expected_probability + "'" +
		      " data-actual='" +
		      htmlEncode(data[i][direction].label_actual) +
		      "' data-actual-prob='" + data[i][direction].label_actual_probability + "'" +
	              " data-token_num='" + i + "'" +
		      ">";
	//}
    	output += htmlEncode(token);
	output += "</span>";

    	if (token === '{') {
    	    output += "\n";
    	    newline = true;
    	    indentation_level += 1;
	} else if (token === '}') {
    	    output += "\n";
    	    newline = true;
	} else if (token === ';') {
    	    output += "\n";
    	    newline = true;
	} else if (alphanumeric(token)) {
	    output += " ";
	}

    }

    $('#linear_heatmap').html(output);
    $('input[name="dir"][value="' + data.direction +'"]').prop('checked', true)
    if (typeof(data.clicked) !== 'undefined') {
        expectation = '';
        d = $("span[data-token_num='" + data.clicked + "']").data();
        for (var k in d) {
            expectation += htmlEncode(k) + ': ' + htmlEncode(d[k]) + '<br>';
        }
        $('#expectation').html(expectation);
    }

    $('#linear_heatmap span').on('click', function(e) {
    	expectation = '';
    	d = $(this).data();
    	for (var k in d) {
    	    expectation += htmlEncode(k) + ': ' + htmlEncode(d[k]) + '<br>';
	}
	probs = data[d.token_num][data.direction]['probabilities']
	for (var i = 0; i < 10; i++) {
	    expectation += htmlEncode(probs[i][1]) + ':' + htmlEncode(probs[i][0]) + '<br>';
	}
	data.clicked = d.token_num;

	$('#expectation').html(expectation);
	$('#expectation').show();//css({top: event.clientY - 100, left: event.clientX}).show();
    });
    $('#linear_heatmap input:radio').on('change', function(e) {
        data.direction = this.value;
        drawLinear(data)
    });
}
