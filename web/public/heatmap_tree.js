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

function drawTree(treeData) {

    // declares a tree layout and assigns the size
    var treemap = d3.tree()
	.size([width, height]);

    //  assigns the data to a hierarchy using parent-child relationships
    var nodes = d3.hierarchy(treeData);

    // maps the node data to the tree layout
    nodes = treemap(nodes);


    // adds the links between the nodes
    var link = g.selectAll(".link")
	.data( nodes.descendants().slice(1))
    .enter().append("path")
	.attr("class", "link")
	.attr("d", function(d) {
	return "M" + d.x + "," + d.y
	    + "C" + d.x + "," + (d.y + d.parent.y) / 2
	    + " " + d.parent.x + "," +  (d.y + d.parent.y) / 2
	    + " " + d.parent.x + "," + d.parent.y;
	});

    // adds each node as a group
    var node = g.selectAll(".node")
	.data(nodes.descendants())
    .enter().append("g")
	.attr("class", function(d) {
	return "node" +
	    (d.children ? " node--internal" : " node--leaf"); })
	.attr("transform", function(d) {
	return "translate(" + d.x + "," + d.y + ")"; });

    // adds the circle to the node
    node.append("circle")
    .attr('class', 'node-circle')
    .attr("r", 10)
    .attr('style', function(d) {
	return 'stroke:#' + gradient(grad, d.data.ratio)
    })
    // 	function() {
    // 	$('#expectation').hide();
    // });

    // adds the text to the node
    node.append("text")
    .attr("dy", ".35em")
    .attr("y", function(d) { return d.children ? -20 : 20; })
    .style("text-anchor", "middle")
    .text(function(d) { return d.data.name; })
    .on('click', function(d) {
        $('#expectation-actual').text(d.data.name);
        $('#expectation-token').text(d.data.expected);
        $('#expectation-ratio').text(d.data.ratio);
        $('#expectation-target-prob').text(d.data.target_probability);
        $('#expectation-max-prob').text(d.data.expected_probability);

        $('#expectation').css({top: d.y-100 , left: d.x - 100 }).show();
    });

    //var tip = d3.tip()
    //.attr('class', 'd3-tip')
    //.offset([-10, 0])
    //.html(function(d) {
    //    return "<strong>Frequency:</strong> <span style='color:red'>" + 3+  "</span>";
    //})
    //node.call(tip)


    //$('.node-circle').hover(function(e) {
    //    $('#expectation-actual').text($(this).data('actual'));
    //    $('#expectation-token').text($(this).data('expected'));
    //    $('#expectation-ratio').text($(this).data('ratio'));
    //    $('#expectation-target-prob').text($(this).data('target'));
    //    $('#expectation-max-prob').text($(this).data('max'));

    //    $('#expectation').css({top: event.clientY - 100, left: event.clientX}).show();
    //}, function() {
    //	$('#expectation').hide();
    //})

}


$(document).ready(function() {

    var socket = io();//'http://robflix.com:3000');
    $('#submit').on('click', function(){
      socket.emit('code', $('#code').text());
      return false;
    });

    socket.on('response', function(response) {
    	console.log(response);
    	if (response.success === false) {
	    $('#heatmap').text('Something went wrong! Maybe your code doesn\'t parse correctly?');
	} else {
	    //generate_heatmap(response.results);
	    drawTree(response.results);
	}
    });

    // set the dimensions and margins of the diagram
    // XXX globals!!
    margin = {top: 40, right: 90, bottom: 50, left: 90},
	width = 1460 - margin.left - margin.right,
	height = 1000 - margin.top - margin.bottom;

    // append the svg obgect to the body of the page
    // appends a 'group' element to 'svg'
    // moves the 'group' element to the top left margin
    svg = d3.select("#tree").append("svg")
	.attr("width", width + margin.left + margin.right)
	.attr("height", height + margin.top + margin.bottom),
	g = svg.append("g")
	.attr("transform",
		"translate(" + margin.left + "," + margin.top + ")");


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
