function drawTree(treeData, redraw) {
    config = treeData.config;
    // set the dimensions and margins of the diagram
    margin = {top: 40, right: 90, bottom: 50, left: 90},
        width =  3500 - margin.left - margin.right,
        height = 1500 - margin.top - margin.bottom;

    // append the svg object
    // appends a 'group' element to 'svg'
    // moves the 'group' element to the top left margin
    svg = d3.select("#heatmap").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom),
        g = svg.append("g")
        .attr("transform",
                "translate(" + margin.left + "," + margin.top + ")");

    // clear the previous tree
    $("svg g").html("")

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
	})
	.attr("style", function(d) {
	    d = d.data[config[0]][config[1]]
	    if (typeof(d.last_sibling) !== 'undefined')
                return "stroke:" + gradient(grad, 1-d.last_sibling.expected);
	    else if (typeof(d.first_sibling) !== 'undefined')
                return "stroke:" + gradient(grad, 1-d.first_sibling.expected);
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
	return 'stroke:' + gradient(grad, d.data[config[0]][config[1]].label_index.ratio);
    })
    // 	function() {
    // 	$('#expectation').hide();
    // });

    // adds the text to the node
    node.append("text")
    .attr("dy", ".35em")
    .attr("y", function(d) { return d.children ? -20 : 30; })
    .style("text-anchor", "middle")
    .text(function(d) { return d.data.attr; })
    .style("stroke", function(d) {
        if (typeof(d.data[config[0]][config[1]].attr_index) !== 'undefined')
            return '#' + gradient(grad, d.data[config[0]][config[1]].attr_index.ratio);
        else
            return '#000000';
    });

    node.append("text")
    .attr("dy", ".35em")
    .attr("y", function(d) { return d.children ? -30 : 20; })
    .style("text-anchor", "middle")
    .text(function(d) { return d.data.label; })
    .on('click', function(d) {
        return redraw(d.data);
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