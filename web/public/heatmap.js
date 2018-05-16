function htmlEncode(value) {
    return $('<div/>').text(value).html();
}

function displayCode(fixed_code) {
    if (fixed_code === false) {
	$('#fixed_code').html('Could not fix code!');
	return;
    } else {
	$('#fixed_code').html(fixed_code);
    }
}

$(document).ready(function() {

    var socket = io();//'http://robflix.com:3000');
    $('#submit').on('click', function(){
        msg = {
	    'code': $('#code').text(),
	    'model': $('input[name=model]:checked', '#submit_code').val()
        }
        socket.emit('code', JSON.stringify(msg));
        return false;
    });

    socket.on('response', function(response) {
    	console.log(response);
	$('#heatmap').html("");
    	if (response.success === false) {
	    $('#heatmap').text('Something went wrong! Maybe your code doesn\'t parse correctly?');
	} else {
	    if (response.model === 'linear') {
	    	heatmap = drawLinear(response.results.linear);
	    } else {
	    	heatmap = drawTree(response.results.ast);
	    }
	    displayCode(response.results.fixed_code);
	}
    });

    // set the dimensions and margins of the diagram
    // XXX globals!!
    margin = {top: 40, right: 90, bottom: 50, left: 90},
	width = 1460 - margin.left - margin.right,
	height = 1000 - margin.top - margin.bottom;

    // append the svg object
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
