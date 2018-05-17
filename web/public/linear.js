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
    output = ''
    for (var i = 0; i < directions.length; i++) {
        output += '<input type="radio" name="dir" value="' + directions[i] +'"> ' + directions[i] + '<br>';
    }
    // XXX fix this mess
    if (typeof(data.direction) === 'undefined') {
        var direction = directions[0]
    } else {
        direction = data.direction
    }

    data.direction = direction;


    for (var i = 0; i < data.length; i++) {
    	var token = data[i][direction].token;

	if (token === '}') {
    	    indentation_level -= 1;
	}

	if (newline) {
	    output += print_indent(indentation_level);
	    newline = false;
	}

        output += "<span style='background-color:"
        output += gradient(grad, data[i][direction].label_index_ratio) + "' data-token_num='" + i + "'>";
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
        prob_str = ''
    	for (var k in data[data.clicked][data.direction]) {
    	    if (k == 'probabilities') {
                probs = data[data.clicked][data.direction]['probabilities']
                for (var i = 0; i < 10; i++) {
                    prob_str += htmlEncode(probs[i][1]) + ' ' + htmlEncode(probs[i][0]) + '<br>';
                }
            } else {
                expectation += htmlEncode(k) + ': ' + htmlEncode(data[data.clicked][data.direction][k]) + '<br>';
            }
	}
	expectation += '<br><br>' + prob_str

	$('#expectation').html(expectation);
	$('#expectation').show();//css({top: event.clientY - 100, left: event.clientX}).show();
    }

    $('#linear_heatmap span').on('click', function(e) {
	data.clicked = $(this).data('token_num');
	drawLinear(data)
    });
    $('#linear_heatmap input:radio').on('change', function(e) {
        data.direction = this.value;
        drawLinear(data)
    });
}
