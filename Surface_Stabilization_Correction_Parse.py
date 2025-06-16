import os
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
import sys

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

def parse_energy_from_line(energy_line):
    """
    Given a comment line (from the last block of an .xyz file) that contains energy info,
    extract and return the energy as a float.

    For example, if energy_line is:
       "i =        1, E =     -4307.5922380587"
    then this function extracts -4307.5922380587.

    """
    # Look for the pattern "E =" followed by a number.
    match = re.search(r'E\s*=\s*([-+]?[0-9]*\.?[0-9]+)', energy_line)
    if match:
        try:
            return float(match.group(1))
        except Exception as e:
            print(f"Error converting energy value '{match.group(1)}': {e}")
            return None
    else:
        # Fallback: try to convert the whole line to a float.
        try:
            return float(energy_line)
        except Exception as e:
            print(f"Could not parse energy from line: {energy_line}. Error: {e}")
            return None


def extract_energy(file_path):
    """
    Convenience function: extracts the last block from the given .xyz file and returns the energy as a float.
    """
    atom_count, energy_line, geom = extract_last_geometry_and_energy(file_path)
    if energy_line is None:
        return None
    return parse_energy_from_line(energy_line)


def parse_identifiers(filename):
    """
    Given a filename such as:
      extracted_group_13_frame_615_group_2.xyz
    or with an optional "-pos-1" suffix (e.g. extracted_group_13_frame_615_group_2-pos-1.xyz),
    extract and return three numbers:
       conformer, frame, and group
    as integers.

    For example, from "extracted_group_13_frame_615_group_2.xyz" it returns (13, 615, 2).
    """
    pattern = re.compile(
        r'^(?:extracted_)?group_(\d+)_frame_(\d+)_group_(\d+)(?:-pos-\d+)?\.xyz$', re.IGNORECASE
    )
    match = pattern.search(filename)
    if match:
        try:
            conf = int(match.group(1))
            frame = int(match.group(2))
            group = int(match.group(3))
            return conf, frame, group
        except Exception as e:
            print(f"Error converting identifiers in {filename}: {e}")
            return None, None, None
    else:
        print(f"Filename does not match expected pattern: {filename}")
        return None, None, None


def process_energy_to_csv(optimized_folder, pos_folder, output_csv):
    """
    For each file in the optimized folder (with names like:
         extracted_group_XX_frame_YYYY_group_ZZ.xyz)
    this script looks for the corresponding "-pos-1" file in the pos folder (with the same
    base name but with "-pos-1" inserted before the extension).

    For each matching pair, it extracts the energy from the last geometry block of both files,
    and writes a CSV file with columns:
         Group, Energy_optimized, Energy_pos, Delta2, Delta1

    Where:
       - Group is the group number extracted from the filename.
       - Delta2 = (Energy_optimized - Energy_optimized_first_group)
       - Delta1 = (Energy_optimized - Energy_optimized_first_group) - (Energy_pos - Energy_pos_first_group)

    The "first group" values are taken from the first row in sorted order.
    """
    optimized_files = [f for f in os.listdir(optimized_folder) if f.lower().endswith('.xyz')]

    data_rows = []

    for opt_file in optimized_files:
        conf, frame, group = parse_identifiers(opt_file)
        if conf is None or frame is None or group is None:
            continue

        # Construct the expected pos filename by inserting "-pos-1" before the extension.
        base, ext = os.path.splitext(opt_file)
        pos_file = base + "-pos-1" + ext

        opt_path = os.path.join(optimized_folder, opt_file)
        pos_path = os.path.join(pos_folder, pos_file)

        energy_opt = extract_energy(opt_path)
        if energy_opt is None:
            print(f"Could not extract energy from optimized file: {opt_file}")
            continue

        if not os.path.exists(pos_path):
            print(f"Corresponding pos file not found: {pos_file}")
            continue

        energy_pos = extract_energy(pos_path)
        if energy_pos is None:
            print(f"Could not extract energy from pos file: {pos_file}")
            continue

        data_rows.append([group, energy_opt, energy_pos])
        print(
            f"Processed Group {group}: {opt_file} -> Energy_optimized = {energy_opt:.10f}, Energy_pos = {energy_pos:.10f}")

    if not data_rows:
        print("No valid data rows found.")
        return

    # Sort rows by Group (first column)
    data_rows.sort(key=lambda row: row[0])

    # Use the first row as the baseline.
    first_energy_opt = data_rows[0][1]
    first_energy_pos = data_rows[0][2]

    # Append two new columns:
    # Delta2 = (Energy_optimized - first_energy_opt)
    # Delta1 = (Energy_optimized - first_energy_opt) - (Energy_pos - first_energy_pos)
    for row in data_rows:
        delta2 = row[1] - first_energy_opt
        delta1 = (row[1] - first_energy_opt) - (row[2] - first_energy_pos)
        row.extend([delta2 * 2625.5, delta1 * 2625.5])

    # Write CSV file.
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Group", "Energy_optimized", "Energy_pos", "Delta2", "Delta1"])
        for row in data_rows:
            csvwriter.writerow(row)

    print(f"CSV file written to {output_csv}")


def plot_diffs(csv_file1, csv_file2):
    # Read the CSV files (they are comma-delimited)
    try:
        df1 = pd.read_csv(csv_file1)
        df2 = pd.read_csv(csv_file2)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        sys.exit(1)

    # Clean up column names (remove accidental whitespace)
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # Ensure required columns exist in both DataFrames.
    for col in ['Group', 'Delta2', 'Delta1']:
        if col not in df1.columns:
            print(f"Error: Column '{col}' not found in file {csv_file1}")
            sys.exit(1)
        if col not in df2.columns:
            print(f"Error: Column '{col}' not found in file {csv_file2}")
            sys.exit(1)

    # Determine the range (number of structure types) for each dataset.
    # We assume Group numbers are numeric.
    min_group1 = df1['Group'].min()
    max_group1 = df1['Group'].max()
    num_groups1 = int(max_group1 - min_group1 + 1)

    min_group2 = df2['Group'].min()
    max_group2 = df2['Group'].max()
    num_groups2 = int(max_group2 - min_group2 + 1)

    # Define a base width (in inches) per group.
    group_width = 0.3  # You may adjust this value for spacing.
    width1 = group_width * num_groups1
    width2 = group_width * num_groups2
    total_width = width1 + width2 + 1  # Add extra space for padding.

    # Create two subplots side by side.
    # Use gridspec_kw to have widths proportional to the number of groups.
    fig, axes = plt.subplots(ncols=2, figsize=(total_width, 6),
                             gridspec_kw={'width_ratios': [width1, width2]},
                             sharey=True)

    # --- Left subplot (CSV file 1) ---
    ax1 = axes[0]
    ax1.plot(df1['Group'], df1['Delta2'], marker='', linestyle='-', color='red',
             label=r'$\Delta E$, kJ/mol')
    ax1.plot(df1['Group'], df1['Delta1'], marker='', linestyle='-', color='blue',
             label=r'$\Delta E$ (surf.-stab.), kJ/mol')

    # Set the x-axis ticks and limits so that spacing between points is uniform.
    ax1.set_xticks(range(int(min_group1), int(max_group1) + 1))
    ax1.set_xlim(min_group1 - 0.5, max_group1 + 0.5)

    # Set axis labels and legend on the left subplot.
    ax1.set_xlabel("Structure type")
    ax1.set_ylabel("Energy, kJ/mol")

    # Remove the frame (top and right spines) for the left subplot.
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend()
    ax1.grid(True)

    # Add title for left subplot.
    # Here, "Bu" appears in superscript and "Cp₂" has the 2 as a subscript.
    ax1.set_title(r"$^{\mathrm{Bu}}\mathrm{Cp}_{2}\mathrm{ZrH@SiOH\text{-}1}$")

    # --- Right subplot (CSV file 2) ---
    ax2 = axes[1]
    ax2.plot(df2['Group'], df2['Delta2'], marker='', linestyle='-', color='red')
    ax2.plot(df2['Group'], df2['Delta1'], marker='', linestyle='-', color='blue')

    ax2.set_xticks(range(int(min_group2), int(max_group2) + 1))
    ax2.set_xlim(min_group2 - 0.5, max_group2 + 0.5)

    # Remove the frame (top and right spines) for the right subplot.
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Only the x-axis label is added for the right subplot.
    ax2.set_xlabel("Structure type")
    ax2.grid(True)

    # Add title for right subplot.
    ax2.set_title(r"$^{\mathrm{Bu}}\mathrm{Cp}_{2}\mathrm{ZrH@SiOH\text{-}2}$")

    plt.tight_layout()
    plt.savefig('surf_stab_figure.png')
    plt.show()

if __name__ == '__main__':
    # Set your folders and output CSV filename:
    optimized_folder = '/Users/akolganov/PycharmProjects/ZrCp2_automatizing/Post_proc_single_Gateway/1C_ZrC_short/Surf_Rel'  # Folder with files like extracted_group_13_frame_615_group_2.xyz
    pos_folder = '/Users/akolganov/PycharmProjects/ZrCp2_automatizing/Post_proc_single_Gateway/1C_ZrC_short/Surf_Rel/surf_stab_opted'  # Folder with files like extracted_group_13_frame_615_group_2-pos-1.xyz
    output_csv = 'surf_stab_1C.csv'

    process_energy_to_csv(optimized_folder, pos_folder, output_csv)

    # Set your folders and output CSV filename:
    optimized_folder = '/Users/akolganov/PycharmProjects/ZrCp2_automatizing/Post_proc_single_Gateway/2C_ZrC_short/Surf_Rel'  # Folder with files like extracted_group_13_frame_615_group_2.xyz
    pos_folder = '/Users/akolganov/PycharmProjects/ZrCp2_automatizing/Post_proc_single_Gateway/2C_ZrC_short/Surf_Rel/surf_stab_opted'  # Folder with files like extracted_group_13_frame_615_group_2-pos-1.xyz
    output_csv = 'surf_stab_2C.csv'

    process_energy_to_csv(optimized_folder, pos_folder, output_csv)

    plot_diffs('surf_stab_1C.csv', 'surf_stab_2C.csv')



