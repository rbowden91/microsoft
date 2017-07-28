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
    var color = '';
    for (var i = 0; i < 3; i++) {
	color += ("00" + grad[idx][i].toString(16)).slice(-2);
    }
    return color;
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

var color1 = [255, 0, 0];
var color2 = [0, 0, 255];
var steps = 16;

var grad = generate_gradient(color1, color2, steps);

function htmlEncode(value) {
    return $('<div/>').text(value).html();
}

function generate_heatmap(data) {
    // XXX for now, very rough guidelines for spacing and indentation. doesn't handle single-line ifs, loops, etc., but
    // those might not exist from pycparser?
    var output = '';
    var indentation_level = 0;
    var newline = false;
    for (var i = 0; i < data.length; i++) {
    	token = data[i].token;

	if (token === '}') {
    	    indentation_level -= 1;
	}

	if (newline) {
	    output += print_indent(indentation_level);
	    newline = false;
	}

	if (i !== 0) {
	    if (data[i][0] === "'")
	    	console.log(data[i])
	    output += "<span style='background-color:#" + gradient(grad, data[i-1].ratio) + "' data-expected=" +
		      (data[i-1].expected === "'" ? "\"'\"" : ("'" + data[i-1].expected + "'")) +
		      " data-ratio='" + data[i-1].ratio + "'" +
		      " data-max='" + data[i-1].expected_probability + "'" +
		      " data-target='" + data[i-1].target_probability + "'" +
		      " data-actual=" +
		      (data[i-1].target === "'" ? "\"'\"" : ("'" + data[i-1].target + "'")) +
		      ">";
	}
    	output += htmlEncode(token);
	if (i !== 0) {
	    output += "</span>";
	}

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

    $('#heatmap').html(output);

    $('#heatmap span').hover(function(e) {
	$('#expectation-actual').text($(this).data('actual'));
	$('#expectation-token').text($(this).data('expected'));
	$('#expectation-ratio').text($(this).data('ratio'));
	$('#expectation-target-prob').text($(this).data('target'));
	$('#expectation-max-prob').text($(this).data('max'));

	$('#expectation').css({top: event.clientY - 100, left: event.clientX}).show();
    }, function() {
    	$('#expectation').hide();
    })
}


$(document).ready(function() {

    var socket = io();//'http://robflix.com:3000');
    $('#submit').on('click', function(){
      socket.emit('code', $('#code').text());
      return false;
    });

    socket.on('response', function(response) {
    	if (response.success === false) {
	    $('#heatmap').text('Something went wrong! Maybe your code doesn\'t parse correctly?');
	} else {
	    generate_heatmap(response.results);
	}
    });

    var start = `
#include <cs50.h> //adds GetString(), which basically renames char to string
#include <stdio.h> //allows things like printftouch
#include <stdlib.h> //adds the atoi() function
#include <string.h> //doin stuff with strings like strlen
#include <ctype.h> //adds isupper(), islower(), and isalpha()

int main(int argc, string argv[])
{
    //defining all my variables
    string keyword;
    string inputtext;
    int j = 0;
    int kwl;

    //test for an argument
    if (argc != 2)
    {
        printf("Try again, gimme one argument of only aplha characters.");
        return 1;
    }

    keyword = argv[1];
    kwl = strlen(keyword);

    //test that argument is all alpha
    for (int i = 0, n = kwl; i < n; i++)
    {
        if (!isalpha(argv[1][i]))
        {
            printf("Try again, gimme letters only.");
            return 1;
        }
        else
        {
            if (isupper(keyword[i]))
            {
                keyword[i] -= 'A';
            }
            else
            {
                keyword[i] -= 'a';
            }
        }
    }

    inputtext = GetString();

    for (int i = 0, n = strlen(inputtext); i < n; i++)
    {
        if (isalpha(inputtext[i]))
        {

            if (isupper(inputtext[i]))
            {
                inputtext[i] = ((((inputtext[i] - 'A') + (keyword[j % kwl])) % 26) +'A');
                j++;
            }
            else
            {
                inputtext[i] = ((((inputtext[i] - 'a') + (keyword[j % kwl])) % 26) + 'a');
                j++;
            }

        }
    }
    printf("%s\\n", inputtext);
}`.trim();

   $('#code').text(start);
});
