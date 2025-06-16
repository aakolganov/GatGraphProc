import os
import re
import graphviz
import pandas as pd
import networkx as nx
from graphviz import Digraph
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from networkx.drawing.nx_pydot import read_dot


# 1. Functions for processing the .xyz and Gateway output files

# Adjusting the script to use version sort (sort -V) for sorting the .xyz files and energy data in the .dat file.
def natural_sort_key(s):
    """ Sort the given list in the increasing order of the integers contained in the string."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def extract_last_geometry_and_energy(file_path):
    """
    extract the last geometry and energy from the .xyz file
    :param file_path: 
    :return: 
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the start of the last geometry block
    start_index = None
    for i in range(len(lines) - 1, -1, -1):
        try:
            int(lines[i].strip())
            start_index = i
            break
        except ValueError:
            continue

    if start_index is None:
        raise ValueError("No geometry block found in the file")

    # Extract the last geometry block
    last_geometry_block = lines[start_index:]
    atom_count = last_geometry_block[0].strip()
    energy_line = last_geometry_block[1].strip() if "E = " in last_geometry_block[1] else ""

    return atom_count, energy_line, last_geometry_block[2:]

def process_xyz_files(source_folder):
    """
    Process XYZ files in the given source folder, extracting the last geometry and energy data.
    Creates a new folder 'extracted_xyzs' containing individual extracted geometries, 
    a concatenated file of all geometries, and an energy data file.

    :param source_folder: Path to the folder containing .xyz files to process
    :return: None
    """
    
    #source_folder = os.getcwd()
    target_folder = os.path.join(source_folder, 'extracted_xyzs')

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    concatenated_geometries = ""
    energy_data = []

    # Sorting filenames using version sort
    sorted_filenames = sorted([f for f in os.listdir(source_folder) if f.endswith(".xyz")], key=natural_sort_key)

    for filename in sorted_filenames:
        source_file = os.path.join(source_folder, filename)
        atom_count, energy_line, last_geometry = extract_last_geometry_and_energy(source_file)

        # Save individual geometry to a file
        individual_file = os.path.join(target_folder, f"extracted_{filename}")
        with open(individual_file, 'w') as file:
            file.write(atom_count + '\n' + "Filename: " + filename + '\n' + ''.join(last_geometry))

        # Add to concatenated geometries
        concatenated_geometries += atom_count + '\n' + "Filename: " + filename + '\n' + ''.join(last_geometry)

        # Collect energy data
        if energy_line:
            energy_data.append(f"{filename}: {energy_line}")

    # Writing concatenated geometries to a single file
    concatenated_file = os.path.join(target_folder, 'concatenated_geometries.xyz')
    with open(concatenated_file, 'w') as file:
        file.write(concatenated_geometries)

    # Writing energy data to .dat file, sorted using version sort
    energy_file = os.path.join(target_folder, 'all_energies.dat')
    with open(energy_file, 'w') as file:
        for entry in sorted(energy_data, key=natural_sort_key):
            file.write(entry + "\n")


def process_file(file_path):
    """
    Process a .dat file containing group and member information, adjusting member numbers, and creating groups.

    The function reads a .dat with two columns (Group and Member),
    increments all member numbers by 1, and creates groups with continuous ranges
    of member numbers where applicable.

    The new format just make it easier to process the data further

    :param file_path: Path to the input file containing group and member data
    :return: Dictionary where keys are group numbers and values are sorted lists of member numbers
    """
    # Load the file into a DataFrame
    df = pd.read_csv(file_path, sep="\s+", header=None, names=['Group', 'Member'])

    # Increase all numbers in the 'Member' column by one
    df['Member'] = df['Member'] + 1

    # Create a dictionary to hold the final groups and their members
    final_groups = {}

    # Iterate through the DataFrame and group members
    for _, row in df.iterrows():
        group = row['Group']
        member = row['Member']

        if group not in final_groups:
            final_groups[group] = set()
        final_groups[group].add(member)

        # Check if the current member is part of a range
        if _ > 0 and df.iloc[_ - 1]['Group'] == group:
            prev_member = df.iloc[_ - 1]['Member']
            if prev_member < member:  # Ensure it's the next in a potential range
                final_groups[group].update(range(prev_member + 1, member))

    # Convert sets to sorted lists
    for group in final_groups:
        final_groups[group] = sorted(final_groups[group])

    return final_groups

def save_to_new_format(processed_data, output_file_path):
    """
    Save processed group data from the initial .dat file to a new one
    """

    with open(output_file_path, 'w') as file:
        for group, members in processed_data.items():
            line = f"{group}. " + " ".join(map(str, members)) + "\n"
            file.write(line)


def energy_data_to_en(filename):


    """
    Writes the energies of each isomer to a new file in .en format.

    """
    with open(filename, "r") as file:
        lines = file.readlines()

    # Extracting conformer number and energy value
    energies = []
    for line in lines:
        parts = line.split(":")
        conformer_num = parts[0].split("_")[1]
        energy = parts[1].split(",")[1].strip().split("=")[1].strip()
        energies.append((conformer_num, energy))

    # Writing simplified format to a new file
    output_file_name = filename.replace(".dat", ".en")
    with open(output_file_name, "w") as file:
        for conformer_num, energy in energies:
            file.write(f"{conformer_num} {energy}\n")

def read_energy_file(energy_file_path):
    """ Reads the energy file and returns a dictionary with conformer number and energy """
    energy_map = {}
    with open(energy_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                conformer, energy = int(parts[0]), float(parts[1])
                energy_map[conformer] = energy * 2625.5  # Convert from Hartree to kJ/mol
    return energy_map

def process_dat_file(dat_file_path):
    """ Reads the .dat file and returns a dictionary with group ID and conformer number """
    groups = {}
    with open(dat_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            group_id_str = parts[0].rstrip('.')  # Remove trailing period before conversion
            group_id = int(group_id_str)
            for conf in parts[1:]:
                groups[int(conf)] = group_id
    return groups

# 2. Functions for creating and processing the transition graph

def calculate_relative_positions(energy_map, reference_conformer=1):
    """ Calculates relative positions of conformers based on the reference conformer """
    reference_energy = energy_map.get(reference_conformer, 0)
    relative_positions = {conf: (energy - reference_energy) for conf, energy in energy_map.items()}
    return relative_positions

def filter_nodes_by_energy(relative_positions, energy_cutoff):
    """ Filters nodes based on energy cutoff """
    return {conf: energy for conf, energy in relative_positions.items() if energy <= energy_cutoff}

def remove_unreachable_and_isolated_nodes(edges, valid_nodes):
    """ Removes unreachable and isolated nodes from the graph """
    reachable = set()
    def dfs(node):
        if node in reachable:
            return
        reachable.add(node)
        for target in edges.get(node, []):
            if target in valid_nodes:
                dfs(target)
    dfs(1)
    return reachable


def create_grouped_graph(relative_positions, groups, gv_file_path, output_gv_path, energy_cutoff):
    """
    Create a directed graph with nodes representing conformers and edges representing transitions.

    The graph is created using graphviz library with nodes colored based on their energy
    (red for positive, blue for negative) and edges labeled with energy differences.
    Only nodes within the specified energy cutoff are included.
    """
    graph = graphviz.Digraph('G', format='gv', engine='neato')
    filtered_positions = filter_nodes_by_energy(relative_positions, energy_cutoff)

    edges = {}
    with open(gv_file_path, 'r') as file:
        for line in file:
            if '->' in line:
                start, end_info = line.split('->')
                start, end = int(start.strip()), int(end_info.split('[')[0].strip())
                if start in filtered_positions and end in filtered_positions:
                    if start not in edges:
                        edges[start] = []
                    edges[start].append(end)

    valid_nodes = remove_unreachable_and_isolated_nodes(edges, filtered_positions)

    for conf, energy in filtered_positions.items():
        if conf in valid_nodes:
            group_id = groups.get(conf, "Unknown")
            color = "red" if energy > 0 else "blue"
            label = f"{group_id}\n{energy:.0f} kJ/mol"
            graph.node(str(conf), label=label, shape="circle", style="filled", fillcolor=color)

    for start, targets in edges.items():
        for end in targets:
            if start in valid_nodes and end in valid_nodes:
                energy_diff = -(filtered_positions.get(start, 0) - filtered_positions.get(end, 0))
                label = f"{energy_diff:.0f}"
                graph.edge(str(start), str(end), label=label)

    graph.render(output_gv_path)

def parse_gv_to_graph(gv_content):
    """ Parses GraphViz content into a NetworkX graph """
    G = nx.DiGraph()
    node_pattern = re.compile(r'(\d+) \[label="(\d+)\n(-?\d+) kJ/mol".*fillcolor=(blue|red).*\]')
    edge_pattern = re.compile(r'(\d+) -> (\d+) \[label=(-?\d+)\]')
    for match in node_pattern.finditer(gv_content):
        node_id, group, energy, color = match.groups()
        G.add_node(int(node_id), group=int(group), energy=int(energy), color=color)
    for match in edge_pattern.finditer(gv_content):
        source, target, energy_diff = match.groups()
        G.add_edge(int(source), int(target), energy_diff=int(energy_diff))
    return G

# Function to identify subgroups within each group
def identify_subgroups(G):
    """ Identifies subgroups within each group and returns a dictionary with group IDs and subgroups """
    subgroups = {}
    for node, data in G.nodes(data=True):
        group = data['group']
        if group not in subgroups:
            subgroups[group] = []
        subgraph_nodes = [n for n, d in G.nodes(data=True) if d['group'] == group]
        subgraph = G.subgraph(subgraph_nodes)
        components = list(nx.connected_components(subgraph.to_undirected()))
        for component in components:
            if component not in subgroups[group]:
                subgroups[group].append(component)
    return subgroups


def simplify_graph(G, subgroups):
    """ Simplifies the graph by merging nodes within each group """
    simplified_G = nx.DiGraph()
    node_map = {}
    for group, group_subgroups in subgroups.items():
        for subgroup in group_subgroups:
            if group == 1:
                representative = 1
            else:
                energies = {node: G.nodes[node]['energy'] for node in subgroup}
                representative = min(energies, key=energies.get)
            if representative not in simplified_G:
                simplified_G.add_node(representative, **G.nodes[representative])
            for node in subgroup:
                node_map[node] = representative
    for u, v, data in G.edges(data=True):
        new_u = node_map[u]
        new_v = node_map[v]
        if new_u != new_v:
            if simplified_G.has_edge(new_u, new_v):
                if data['energy_diff'] < simplified_G[new_u][new_v]['energy_diff']:
                    simplified_G[new_u][new_v]['energy_diff'] = data['energy_diff']
            else:
                simplified_G.add_edge(new_u, new_v, **data)
    return simplified_G


def reduce_group_transitions(simplified_G, source_node=1):
    """
    Reduce multiple transitions from Node 1 to group nodes (e.g., 14).
    Keep only the transition to the node with the lowest energy if all nodes in the group
    have no outgoing transitions except back to Node 1. Remove nodes for which edges were removed.
    """
    # Get all successors of source_node
    successors = list(simplified_G.successors(source_node))

    # Filter nodes with no outgoing transitions except back to source_node
    terminal_nodes = [
        node for node in successors
        if all(target == source_node for target in simplified_G.successors(node))
    ]

    # Group terminal nodes by their group label
    group_map = {}
    for node in terminal_nodes:
        group = simplified_G.nodes[node]['group']
        if group not in group_map:
            group_map[group] = []
        group_map[group].append(node)

    # Process each group independently
    for group, nodes in group_map.items():
        if len(nodes) > 1:  # Only reduce if multiple nodes exist in the same group
            # Find the node with the lowest energy
            node_with_lowest_energy = min(
                nodes, key=lambda node: simplified_G.nodes[node]['energy']
            )

            # Remove all other nodes and their transitions
            for node in nodes:
                if node != node_with_lowest_energy:
                    # Remove edges to/from the node
                    if simplified_G.has_edge(source_node, node):
                        simplified_G.remove_edge(source_node, node)
                    if simplified_G.has_edge(node, source_node):
                        simplified_G.remove_edge(node, source_node)
                    # Remove the node itself
                    simplified_G.remove_node(node)

    return simplified_G

def recalculate_edge_energy_diffs_with_sign(G):
    """Recalculates the energy difference for each edge based on the sign of the energy difference"""
    for u, v, data in G.edges(data=True):
        energy_u = G.nodes[u]['energy']
        energy_v = G.nodes[v]['energy']
        # Calculate the energy difference with the correct sign
        data['energy_diff'] = energy_v - energy_u

def create_graphviz(simplified_G):
    """Function to create a Graphviz dot format from a NetworkX graph"""
    dot = Digraph(comment='Simplified Molecular Transition Graph')
    for node, data in simplified_G.nodes(data=True):
        label = f"{data['group']}\n{data['energy']} kJ/mol"
        dot.node(str(node), label, fillcolor='blue', style='filled', shape='circle')
    for u, v, data in simplified_G.edges(data=True):
        dot.edge(str(u), str(v), label=str(data['energy_diff']))
    return dot


def process_and_update_graph_with_sign(file_path):
    """Combined function to process, recalculate, and update a graph"""
    with open(file_path, 'r') as file:
        gv_content = file.read()
    G = parse_gv_to_graph(gv_content)

    # Identifying subgroups and simplifying the graph
    subgroups = identify_subgroups(G)
    simplified_G = simplify_graph(G, subgroups)

    # Reduce transitions from the source node (Node 1)
    simplified_G = reduce_group_transitions(simplified_G)

    # Recalculating energy differences with sign
    recalculate_edge_energy_diffs_with_sign(simplified_G)

    # Generating and saving the updated graph
    updated_graphviz_output = create_graphviz(simplified_G)
    output_graph = file_path[:-3] + '_final.gv'
    updated_graphviz_output.save(output_graph)

#3. Visualization

def adjust_font_color_for_custom_cmap(rgb_color):
    """Adjusts the font color for a given RGB color based on the luminance of the color."""
    luminance = 0.299 * rgb_color[0] + 0.587 * rgb_color[1] + 0.114 * rgb_color[2]
    return 'white' if luminance < 0.5 else 'black'


def process_gv_files(gv_file_paths, output_svg_paths, output_heatmap_paths, font_size=24):
    """Plot graphs from Graphviz files and save them as SVG files."""
    # Custom colormap
    cmap_custom = plt.get_cmap('RdYlGn_r')

    for gv_file_path, svg_path, heatmap_path in zip(gv_file_paths, output_svg_paths, output_heatmap_paths):
        # Read the .gv file into a NetworkX graph
        G = read_dot(gv_file_path)

        # Extract energy values from node labels (assuming label is something like "X\n12.34 kJ/mol")
        energy_values = {
            node: float(data['label'].split('\n')[1].split(' ')[0])
            for node, data in G.nodes(data=True)
        }
        min_energy, max_energy = min(energy_values.values()), max(energy_values.values())
        energy_norm = {
            node: (energy - min_energy) / (max_energy - min_energy)
            for node, energy in energy_values.items()
        }

        # Create a Digraph for Graphviz
        dot = Digraph()

        # Add nodes with color and font adjustments
        for node, data in G.nodes(data=True):
            normalized_energy = energy_norm[node]
            # Convert the normalized value to an RGB color using the custom colormap
            rgb_color = cmap_custom(normalized_energy)[:3]
            # Convert to hex
            hex_color = '#' + ''.join(f"{int(c * 255):02x}" for c in rgb_color)
            # Determine appropriate font color (white or black)
            font_color = adjust_font_color_for_custom_cmap(rgb_color)
            # Remove surrounding quotes from label
            label_text = data['label'][1:-1] if data['label'].startswith('"') else data['label']

            dot.node(
                node,
                label=label_text,
                style="filled",
                fillcolor=hex_color,
                fontsize=str(font_size),
                fontcolor=font_color,
                shape="circle"
            )

        # Add edges with thicker lines (penwidth) and larger arrowheads (arrowsize)
        for source, target, data in G.edges(data=True):
            dot.edge(
                source,
                target,
                arrowsize='4.0',  # Increase arrowhead size
                penwidth='1.5',  # Thicker line
                fontsize='72',
                **data
            )

        # Render the graph to SVG format
        dot.format = 'svg'
        dot.render(svg_path.replace('.svg', ''), cleanup=True)

        # Create a heatmap for the energy scale
        fig, ax = plt.subplots(figsize=(1, 8))
        norm = Normalize(vmin=min_energy, vmax=max_energy)
        sm = ScalarMappable(cmap=cmap_custom, norm=norm)
        plt.colorbar(sm, cax=ax, orientation='vertical')
        ax.set_ylabel('Energy (kJ/mol)')
        fig.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)



if __name__ == "__main__":
    # Process the Gateway and CP2K outs
    process_xyz_files('xyz_opted/1C_allxyz')
    file_path_1c = 'Post_proc_single_Gateway/1C/concatenated_geometries_periods.dat'
    processed_data = process_file(file_path_1c)
    output_file_path = 'Post_proc_single_Gateway/1C/processed_concatenated_geometries_periods_1C.dat'
    save_to_new_format(processed_data, output_file_path)
    energy_data_to_en('Post_proc_single_Gateway/1C/all_energies.dat')

    # Crate .dot Graphviz graphs
    energy_cutoff = 100
    dat_file_path = 'Post_proc_single_Gateway/1C/processed_concatenated_geometries_periods_1C.dat'
    gv_file_path = 'transition_graph_1C.gv'
    energy_file_path = 'Post_proc_single_Gateway/1C/all_energies.en'
    output_gv_path_1C = 'Post_proc_single_Gateway/1C/grouped_output_gv_file_1C.gv'
    energy_map = read_energy_file(energy_file_path)
    groups = process_dat_file(dat_file_path)
    relative_positions = calculate_relative_positions(energy_map)
    create_grouped_graph(relative_positions, groups, gv_file_path, output_gv_path_1C, energy_cutoff)
    process_and_update_graph_with_sign(output_gv_path_1C)

    # Plot nice .svg file for the visualization
    final_graph_1C = ['Post_proc_single_Gateway/1C/grouped_output_gv_file_1C_final.gv']
    output_svg_path_1C=['Visualization/1C_graph.svg']
    output_heatmap_paths=['Visualization/1C_heatmap.png']
    process_gv_files(final_graph_1C, output_svg_path_1C, output_heatmap_paths, font_size=48)


    



