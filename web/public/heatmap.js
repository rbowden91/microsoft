function htmlEncode(value) {
    return $('<div/>').text(value).html();
}

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
    if (level > 1) {
        console.log("level > 1", level);
        level = Math.min(level, 1)
    }
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

function recursive_print(d, indent) {
    var expectation = '';
    var prob_str = '';
    for (var k in d) {
        if (k == 'probabilities') {
            probs = d[k];
            for (var i = 0; i < 10; i++) {
                prob_str += htmlEncode(probs[i][1]) + ' ' + htmlEncode(probs[i][0]) + '<br>';
            }
            prob_str += '<br>';
        } else if (typeof(d[k]) === 'object') {
            if (k == 'children') continue;
            expectation += print_indent(indent) + k + ': {<br>';
            rp = recursive_print(d[k], indent + 1);
            expectation += rp[0] + '<br>' + print_indent(indent) + '}';
            prob_str += rp[1];
        } else if (Array.isArray(d[k])) {
            expectation += print_indent(indent) + '[<br>';
            rp = recursive_print(d[k], indent + 1);
            expectation += rp[0] + '<br>' + print_indent(indent) + ']';
            prob_str += rp[1];
        } else {
            expectation += print_indent(indent) + htmlEncode(k) + ': ' + htmlEncode(d[k]) + '<br>';
        }
    }
    return [expectation, prob_str];
}

function draw(data, model, node) {
    configs = Object.keys(data).filter(key => ['joint_configs', 'dependency_configs'].includes(key))

    var output = '';
    for (var i = 0; i < configs.length; i++) {
        for (var config in data[configs[i]]) {
            var found_config = [configs[i], config];
            var name = configs[i] + ' ' + config;
            output += '<input type="radio" data-config-type="' + configs[i] + '" + data-config-name="' + config +'" name="dir" id="' + name + '"><label for="' + name +'"> ' + name + '</label><br>';
        }
    }
    // XXX fix this mess
    if (typeof(data.config) === 'undefined') {
        data.config = found_config;
    }


    if (model == 'ast') {
        $('#heatmap').html(output);
        output = drawTree(data, function(node_number) { draw(data, model, node_number); })
    } else {
        output += drawLinear(data)
        $('#heatmap').html(output);
    }

    if (typeof(node) !== 'undefined') {
        data.clicked = node;
    }

    $('input[name="dir"][data-config-type="' + data.config[0] +'"][data-config-name="' + data.config[1] +'"]').prop('checked', true)
    if (typeof(data.clicked) !== 'undefined') {
        if (model == 'linear') {
            d = data[data.clicked][data.config[0]][data.config[1]];
        } else {
            d = data.clicked[data.config[0]][data.config[1]];
        }
        console.log(data.config[0], data.config[1])
        var rp = recursive_print(d, 1);
        rp[0] = data.config[0] + ' ' + data.config[1] + ' = {<br>' + rp[0] + '<br>}';
        console.log(rp)
	expectation = rp[0] + '<br><br>' + rp[1];

	$('#expectation').html(expectation);
	$('#expectation').show();//css({top: event.clientY - 100, left: event.clientX}).show();
    }

    $('#heatmap span').on('click', function(e) {
	data.clicked = $(this).data('token_num');
	draw(data, model, data.clicked)
    });
    $('#heatmap input:radio').on('change', function(e) {
        data.config = [$(this).data('config-type'), $(this).data('config-name')]
        draw(data, model, data.clicked)
    });
}

function displayCode(fixed_code) {
    if (fixed_code === false) {
	$('#fixed_code').html('Could not fix code!');
	return;
    } else {
	$('#fixed_code').html(fixed_code);
    }
}
