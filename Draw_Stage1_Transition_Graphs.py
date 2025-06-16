from collections import defaultdict
import os as os

def parse_conf_file_for_gv(conf_file):
    """
    Parses a .conf file to extract frame and group data.
    Args:
        conf_file (str): Path to the .conf file.
    Returns:
        list: List of (frame, group) tuples.
    """
    with open(conf_file, 'r') as file:
        lines = file.readlines()

    # Extract frame and group data from the file
    frames = []
    for line in lines[3:]:  # Skip the first three metadata lines
        frame, group = map(int, line.split())
        frames.append((frame, group))

    return frames


def generate_transition_graph(conf_directory, output_file):
    """
    Generates a transition graph in Graphviz .gv format.
    Args:
        conf_directory (str): Directory containing .conf files.
        output_file (str): Path to the output .gv file.
    """
    transitions = defaultdict(int)

    # Process each .conf file
    for conf_file in os.listdir(conf_directory):
        if conf_file.endswith('.conf'):
            conf_path = os.path.join(conf_directory, conf_file)
            frames = parse_conf_file_for_gv(conf_path)

            # Analyze consecutive group transitions
            for i in range(1, len(frames)):
                group_from = frames[i - 1][1]
                group_to = frames[i][1]
                if group_from != group_to:
                    transitions[(group_from, group_to)] += 1

    # Write the Graphviz .gv file
    with open(output_file, 'w') as file:
        file.write('digraph G {\n')
        file.write('label="Graph with frequencies";\n')
        file.write('node [style=filled];\n')
        file.write('graph [bgcolor=transparent];\n')
        file.write('node [shape = circle];\n')

        # Write nodes and edges
        nodes = set()
        for (group_from, group_to), frequency in transitions.items():
            nodes.add(group_from)
            nodes.add(group_to)
            file.write(f'  {group_from}-> {group_to} [label="{frequency}"];\n')

        # Define node properties
        for node in nodes:
            file.write(f'  {node} [fillcolor=yellow];\n')

        file.write('}\n')

    print(f"Transition graph saved to {output_file}")

if __name__ == "__main__":
    conf_directory = "./conf_files/1C"  # Directory containing .conf files
    output_file = "transition_graph_1C.gv"  # Output .gv file

    generate_transition_graph(conf_directory, output_file)