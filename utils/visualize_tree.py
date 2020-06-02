import os, sys

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import pickle

def construct_tree_for_visual(tree, node_info_key, depth=0):
    tree_for_visual = {'smiles': 'http://askcos.mit.edu/draw/smiles/' + str(tree['smiles']).replace('#', '%23'),
                       'depth': depth,
                       'children': []}

    if node_info_key in tree.keys():
        if type(tree[node_info_key]) is not str:
            tree_for_visual['score'] = '{:.3f}'.format(tree[node_info_key])
        else:
            tree_for_visual['score'] = tree[node_info_key]
    else:
        tree_for_visual['score'] = ''

    if tree['child']:
        # tree_for_visual['children'] = []
        for child in tree['child']:
            tree_for_visual['children'].append(construct_tree_for_visual(child, node_info_key, depth+1))
    return tree_for_visual


def construct_tree_for_d3_visualization(tree, depth, new_tree={}, max_children=0):
    # if 'is_chemical' in tree.keys():

    new_tree['smiles'] = 'http://askcos.mit.edu/draw/smiles/' + str(tree['smiles']).replace('#', '%23')
    if 'score' in tree.keys():
        new_tree['score'] = str(tree['score'])
    else:
        new_tree['score'] = ''
    # new_tree['smiles'] = str(new_tree['smiles'])

    new_tree['children'] = []
    if tree['child']:
        # print(len(tree['child']))
        if max_children < len(tree['child']):
            max_children = len(tree['child'])
        for child in tree['child']:
            new_tree['children'].append({})
            _, max_children = construct_tree_for_d3_visualization(child, depth + 1, new_tree['children'][-1], max_children)
    return new_tree, max_children

def count_tree_depth_children(tree, count):
    count[tree['depth']] += 1
    if tree['children']:
        for child in tree['children']:
            count = count_tree_depth_children(child, count)
    return count

def create_tree_html(trees, file_name, tree_info=None, node_info_key='score',
                     width_factor=1, max_depth=10):
    try:
        outfile = file_name

    except Exception as e:
        print(e)
        print('Need to specify file name to write results to')

    trees_for_visualization = {'name': 'dummy_root', 'children': []}
    max_children = 1
    for i, tree in enumerate(trees):
        output = construct_tree_for_visual(tree, node_info_key)
        trees_for_visualization['children'].append(output)
        # print()
        current_children = max(count_tree_depth_children(output, count=[0]*20))
        if current_children > max_children:
            max_children = current_children
        if tree_info:
            # print(tree_info[i])
            trees_for_visualization['children'][-1]['_id'] = tree_info[i]
        else:
            trees_for_visualization['children'][-1]['_id'] = ('T%d' % i)
    # print(max_children)
    max_children = max(max_children, 3)
    height = 300 * len(trees) * max_children / 3 * width_factor
    page_width = max_depth * 300

    fid_out = open(outfile + '.html', 'w')
    fid_out.write('<!DOCTYPE html>\n')
    fid_out.write('  <head>\n')
    fid_out.write('    <meta charset="utf-8">\n')
    fid_out.write('    <title>{}</title>\n'.format(outfile))
    fid_out.write('    <style>\n')
    fid_out.write(' .node circle {\n')
    fid_out.write('   fill: #fff;\n')
    fid_out.write('   stroke: steelblue;\n')
    fid_out.write('   stroke-width: 3px;\n')
    fid_out.write(' }\n')
    fid_out.write(' .node rect {\n')
    fid_out.write('     fill: #fff;\n')
    fid_out.write('     stroke: steelblue;\n')
    fid_out.write('     stroke_width: 3px;\n')
    fid_out.write(' }\n')
    fid_out.write(' .node text { font: 12px sans-serif; }\n')
    fid_out.write(' .link {\n')
    fid_out.write('   fill: none;\n')
    fid_out.write('   stroke: #ccc;\n')
    fid_out.write('   stroke-width: 2px;\n')
    fid_out.write(' }\n')
    fid_out.write('    </style>\n')
    fid_out.write('  </head>\n')
    fid_out.write('  <body>\n')
    fid_out.write('<!-- load the d3.js library -->  \n')
    fid_out.write('<script src="http://d3js.org/d3.v3.min.js"></script>\n')
    fid_out.write('<script>\n')
    fid_out.write('var treeData = [\n')
    fid_out.write('{}\n'.format(trees_for_visualization))
    fid_out.write('];\n')
    fid_out.write('var margin = {top: 20, right: 120, bottom: 20, left: 0},\n')
    fid_out.write(' width = {} - margin.right - margin.left,\n'.format(page_width))
    fid_out.write(' height = {} - margin.top - margin.bottom;\n'.format(height))
    fid_out.write('var i = 0;\n')
    fid_out.write('var tree = d3.layout.tree()\n')
    fid_out.write(' .size([height, width]);\n')
    fid_out.write('var diagonal = d3.svg.diagonal()\n')
    fid_out.write(' .projection(function(d) { return [d.y, d.x]; });\n')
    fid_out.write('var svg = d3.select("body").append("svg")\n')
    fid_out.write(' .attr("width", width + margin.right + margin.left)\n')
    fid_out.write(' .attr("height", height + margin.top + margin.bottom)\n')
    fid_out.write('  .append("g")\n')
    fid_out.write(' .attr("transform", \n')
    fid_out.write('       "translate(" + margin.left + "," + margin.top + ")");\n')
    fid_out.write('root = treeData[0];\n')
    fid_out.write('update(root);\n')
    fid_out.write('function update(source) {\n')
    fid_out.write('  // Compute the new tree layout.\n')
    fid_out.write('  var nodes = tree.nodes(root).reverse(),\n')
    fid_out.write('   links = tree.links(nodes);\n')
    fid_out.write('  // Normalize for fixed-depth.\n')
    fid_out.write('  nodes.forEach(function(d) { d.y = d.depth * 250; });\n')
    fid_out.write('  // Declare the nodes…\n')
    fid_out.write('  var node = svg.selectAll("g.node")\n')
    fid_out.write('   .data(nodes, function(d) { return d.id || (d.id = ++i); });\n')
    fid_out.write('  // Enter the nodes.\n')
    fid_out.write('  var nodeEnter = node.enter().append("g")\n')
    fid_out.write('   .attr("class", "node")\n')
    fid_out.write('   .attr("transform", function(d) { \n')
    fid_out.write('       return "translate(" + d.y + "," + d.x + ")"; });\n')
    fid_out.write('  nodeEnter.append("image")\n')
    fid_out.write('      .attr("xlink:href", function(d) { return d.smiles; })\n')
    fid_out.write('      .attr("x", "-60px")\n')
    fid_out.write('      .attr("y", "-60px")\n')
    fid_out.write('      .attr("width", "120px")\n')
    fid_out.write('      .attr("height", "120px");\n')
    fid_out.write('  nodeEnter.append("path")\n')
    fid_out.write('       .style("stroke", "black")\n')
    fid_out.write('       .style("fill", function(d) { if (d.freq==1) { return "white"; }\n')
    fid_out.write('                                     else if (d.freq==2) { return "yellow";}\n')
    fid_out.write('                                     else if (d.freq==3) { return "orange"; }\n')
    fid_out.write('                                     else if (d.freq>=4) { return "red"; }\n')
    fid_out.write('                                     else {return "white";}\n')
    fid_out.write('                                     })\n')
    fid_out.write('       .attr("d", d3.svg.symbol()\n')
    fid_out.write('                     .size(0)\n')
    fid_out.write('                     .type(function(d) {if\n')
    fid_out.write('                         (d.rc_type == "chemical") {return "circle";} else if\n')
    fid_out.write('                         (d.rc_type == "reaction") {return "cross";}\n')
    fid_out.write('                     }));\n')
    fid_out.write('  nodeEnter.append("text")\n')
    fid_out.write('   .attr("x", 0)\n')
    fid_out.write('   .attr("y", 35)\n')
    fid_out.write('   .attr("text-anchor", function(d) { \n')
    fid_out.write('       return d.children || d._children ? "end" : "start"; })\n')
    fid_out.write('   .text(function(d) { return d.names; })\n')
    fid_out.write('   .style("fill-opacity", 1);\n')
    fid_out.write('  nodeEnter.append("text")\n')
    fid_out.write('   .attr("x", 200)\n')
    fid_out.write('   .attr("y", 120)\n')
    fid_out.write('   .attr("text-anchor", function(d) { \n')
    fid_out.write('       return d.children || d._children ? "end" : "start"; })\n')
    fid_out.write('   .text(function(d) { return d._id; })\n')
    fid_out.write('   .style("fill-opacity", 1);\n')
    fid_out.write('  nodeEnter.append("text")\n')
    fid_out.write('   .attr("x", 0)\n')
    fid_out.write('   .attr("y", -30)\n')
    fid_out.write('   .attr("text-anchor", function(d) { \n')
    fid_out.write('       return d.children || d._children ? "end" : "start"; })\n')
    fid_out.write('   .text(function(d) { return d.score; })\n')
    fid_out.write('   .style("fill-opacity", 1);\n')
    fid_out.write('  // Declare the links…\n')
    fid_out.write('  var link = svg.selectAll("path.link")\n')
    fid_out.write('   .data(links, function(d) { return d.target.id; });\n')
    fid_out.write('  // Enter the links.\n')
    fid_out.write('  link.enter().insert("path", "g")\n')
    fid_out.write('   .attr("class", "link")\n')
    fid_out.write('   .style("stroke", function(d) { return d.target.level; })\n')
    fid_out.write('   .attr("d", diagonal);\n')
    fid_out.write('  // remove the first level, leaving the targets as the first level\n')
    fid_out.write('  node.each(function(d){\n')
    fid_out.write(' if (d.name == "dummy_root")\n')
    fid_out.write('     d3.select(this).remove();});\n')
    fid_out.write(' link.each(function(d){\n')
    fid_out.write(' if (d.source.name == "dummy_root") \n')
    fid_out.write('     d3.select(this).remove();});\n')
    fid_out.write('}\n')
    fid_out.write('</script>\n')
    fid_out.write('  </body>\n')
    fid_out.write('</html>\n')

    fid_out.close()


if __name__ == "__main__":
    file_name = project_path + '/data/pathway_train_example.pkl'
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    trees_to_plot = [d['tree'] for d in data['generated_paths'][0:10]]
    create_tree_html(trees_to_plot, 'plotted_trees')
