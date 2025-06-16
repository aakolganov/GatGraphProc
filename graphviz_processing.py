import networkx as nx
from networkx.drawing.nx_agraph import read_dot, graphviz_layout
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pygraphviz as pgv
from IPython.display import Image
import numpy as np

# Step 1: Convert Graphviz .gv file to a NetworkX graph
gv_file_path = 'path/to/your/file.gv'  # Replace with your actual file path
G = read_dot(gv_file_path)

# Step 2: Extract energy values from node labels and normalize for color mapping
energy_values = {node: float(data['label'].split('\n')[1].split(' ')[0]) for node, data in G.nodes(data=True)}
min_energy = min(energy_values.values())
max_energy = max(energy_values.values())
energy_norm = {node: (energy - min_energy) / (max_energy - min_energy) for node, energy in energy_values.items()}

# Step 3: Convert the NetworkX graph to a PyGraphviz AGraph
A = nx.nx_agraph.to_agraph(G)

# Step 4: Apply color coding based on energy and increase font size to 18
colormap = plt.get_cmap('viridis')
for node in A.nodes():
    normalized_energy = energy_norm[node.get_name()]
    hex_color = colormap(normalized_energy)[:3]  # RGB to HEX conversion handled internally
    node.attr['style'] = 'filled'
    node.attr['fillcolor'] = '#'+"".join("%02x"%int(c*255) for c in hex_color)
    node.attr['fontsize'] = 18

# Step 5: Layout and render the graph with PyGraphviz
A.layout(prog='dot')
output_path = '/mnt/data/graph_dot_layout_final.png'
A.draw(output_path)

# Step 6: Combine the graph image with a heatmap using Matplotlib
graph_img = plt.imread(output_path)
fig, ax = plt.subplots(figsize=(12, 8))
img_ax = fig.add_axes([0.05, 0.1, 0.6, 0.8], frame_on=False)
cbar_ax = fig.add_axes([0.67, 0.1, 0.03, 0.8], frame_on=False)

img_ax.imshow(graph_img)
img_ax.set_xticks([])
img_ax.set_yticks([])
img_ax.axis('off')

norm = Normalize(vmin=min_energy, vmax=max_energy)
sm = ScalarMappable(cmap='viridis', norm=norm)
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
cbar_ax.set_ylabel('Energy (kJ/mol)')
cbar_ax.yaxis.set_ticks_position('left')
cbar.outline.set_visible(False)

plt.show()
