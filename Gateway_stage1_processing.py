import os
import pandas as pd


def parse_conf_file(conf_file):
    """
    Parses a .conf file from Gateway to extract metadata and frame data.

    A .conf file contains the following information:
    Header (trajectory file name)
    Total number of frames
    Total number of atoms
    Then for each isomers: frame number when this isomer appears, group number

    E.g.: \n
    0 	 1 \n
    870 	 164 \n
    958 	 1 \n
    This would mean that from the frame 0 to frame 869 there is isomer from group 1,
    then from 870 to 957 there is isomer from group 164, and finally from 958 to the end there is isomer from group 1.

    Args:
        conf_file (str): Path to the .conf file.
    Returns:
        dict: Parsed data with metadata and frames.
    """
    with open(conf_file, 'r') as file:
        lines = file.readlines()

    data = {
        'xyz_file': lines[0].strip(),
        'total_frames': int(lines[1].strip()),
        'total_groups': int(lines[2].strip()),
        'frames': []
    }

    for line in lines[3:]:
        frame, group = map(int, line.split())
        data['frames'].append((frame, group))

    return data

def calculate_all_periods(data):
    """
    Calculates all continuous periods for each isomer group.
    Args:
        data (dict): Parsed data from a .conf file.
    Returns:
        pd.DataFrame: DataFrame with all periods for each group.
    """
    frames_df = pd.DataFrame(data['frames'], columns=['Frame', 'Group'])
    frames_df = frames_df.sort_values('Frame').reset_index(drop=True)

    periods = []
    current_group = None
    current_start = None

    for _, row in frames_df.iterrows():
        frame, group = row['Frame'], row['Group']

        if current_group is None or group != current_group:
            # Finalize the current period
            if current_group is not None:
                periods.append({
                    'Group': current_group,
                    'Start Frame': current_start,
                    'End Frame': frame - 1,
                    'Period Length': frame - current_start,
                    'Source File': data['xyz_file']
                })
            # Start a new period
            current_group = group
            current_start = frame

    # Finalize the last period
    if current_group is not None:
        periods.append({
            'Group': current_group,
            'Start Frame': current_start,
            'End Frame': frames_df['Frame'].iloc[-1],
            'Period Length': frames_df['Frame'].iloc[-1] - current_start + 1,
            'Source File': data['xyz_file']
        })

    return pd.DataFrame(periods)

def process_multiple_conf_files(conf_directory, output_csv):
    """
    Processes multiple .conf files and saves all periods into a single CSV file.
    Args:
        conf_directory (str): Directory containing .conf files.
        output_csv (str): Path to the output CSV file.
    """
    all_results = []

    for conf_file in os.listdir(conf_directory):
        if conf_file.endswith('.conf'):
            conf_path = os.path.join(conf_directory, conf_file)
            parsed_data = parse_conf_file(conf_path)
            results = calculate_all_periods(parsed_data)
            all_results.append(results)

    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(output_csv, index=False)
    print(f"All periods saved to {output_csv}")

def find_longest_period(all_periods_csv, output_csv):
    """
    Finds the longest period for each conformer from the all_periods CSV file.
    Args:
        all_periods_csv (str): Path to the CSV file containing all periods.
        output_csv (str): Path to save the longest periods CSV file.
    """
    all_periods = pd.read_csv(all_periods_csv)
    longest_periods = all_periods.loc[
        all_periods.groupby('Group')['Period Length'].idxmax()
    ].reset_index(drop=True)

    longest_periods.to_csv(output_csv, index=False)
    print(f"Longest periods saved to {output_csv}")


def extract_frame_xyz(xyz_file, frame_number):
    """
    Extracts the geometry of a specific frame from an .xyz file.
    Args:
        xyz_file (str): Path to the .xyz file.
        frame_number (int): Frame number to extract.
    Returns:
        str: Extracted geometry as a string.
    """
    with open(xyz_file, 'r') as file:
        lines = file.readlines()

    frame_indicator = f"i = {frame_number},"
    frame_start = False
    extracted_frame = []

    for line in lines:
        # Detect the start of the desired frame
        if frame_indicator in ' '.join(line.split()):
            frame_start = True
            continue  # Skip the frame header line

        # Collect atom data until the next frame or atom count
        if frame_start and not line.strip().isdigit() and "i =" not in line:
            extracted_frame.append(line)
        elif frame_start and (line.strip().isdigit() or "i =" in line):
            break  # Stop collecting when a new frame starts

    # Check if the frame was successfully extracted
    if not extracted_frame:
        raise ValueError(f"Frame {frame_number} not found or incomplete in {xyz_file}")

    return ''.join(extracted_frame)

def generate_cp2k_input(xyz_geometry, cp2k_template, project_name):
    """
    Generates a CP2K input file content using the geometry and a template.
    Args:
        xyz_geometry (str): XYZ geometry string.
        cp2k_template (str): CP2K input template string.
        project_name (str): Name of the CP2K project.
    Returns:
        str: Complete CP2K input file content.
    """
    return cp2k_template.format(coordinates=xyz_geometry, project_name=project_name)


def process_longest_periods(longest_periods_csv, xyz_directory, output_directory, cp2k_template):
    """
    Processes the longest periods and generates CP2K input files.
    Args:
        longest_periods_csv (str): Path to the CSV file containing the longest periods.
        xyz_directory (str): Directory containing the .xyz trajectory files.
        output_directory (str): Directory to save generated input files.
        cp2k_template (str): Template for the CP2K input files.
    """
    os.makedirs(output_directory, exist_ok=True)

    longest_periods = pd.read_csv(longest_periods_csv)

    for _, row in longest_periods.iterrows():
        group = row['Group']
        start_frame = row['Start Frame']
        end_frame = row['End Frame']
        source_file = row['Source File']
        mid_frame = start_frame + (end_frame - start_frame) // 2

        # Locate the corresponding .xyz file
        xyz_file = os.path.join(xyz_directory, source_file)
        if not os.path.exists(xyz_file):
            print(f"XYZ file not found: {xyz_file}")
            continue

        try:
            geometry = extract_frame_xyz(xyz_file, mid_frame)
            project_name = f"group_{group}_frame_{mid_frame}"
            cp2k_input_content = generate_cp2k_input(geometry, cp2k_template, project_name)

            output_file = os.path.join(output_directory, f"{project_name}.inp")
            with open(output_file, 'w') as file:
                file.write(cp2k_input_content)

            print(f"Generated CP2K input file: {output_file}")

        except ValueError as e:
            print(f"Error processing group {group}: {e}")


# CP2K input template
cp2k_template = """
&GLOBAL
   PRINT_LEVEL  MEDIUM
   PROJECT_NAME {project_name}
   RUN_TYPE  GEO_OPT
&END GLOBAL

&FORCE_EVAL
   METHOD QS
   &DFT
     LSD  TRUE
 !    MULTIPLICITY 1
     CHARGE 0
     BASIS_SET_FILE_NAME path/to/cp2k-2022.2/data/BASIS_MOLOPT_UCL
     BASIS_SET_FILE_NAME path/to/cp2k-2022.2/data/BASIS_MOLOPT
     POTENTIAL_FILE_NAME path/to/cp2k-2022.2/data/GTH_POTENTIALS 
     &MGRID
       CUTOFF 450
       REL_CUTOFF 30
       NGRIDS 4
     &END MGRID

     &QS
       METHOD  GPW
     &END QS

     &SCF
      !MAX_SCF          1             ! Max n of iterations
       EPS_SCF          0.1E-04       ! SCF converergence
       SCF_GUESS        ATOMIC
       &OT
         MINIMIZER DIIS
         PRECONDITIONER FULL_SINGLE_INVERSE
       &END OT
     &END SCF

     &XC 
       &XC_FUNCTIONAL PBE
       &END XC_FUNCTIONAL
      ! &XC_GRID
       !  XC_DERIV SPLINE2
       !  XC_SMOOTH_RHO NONE
      ! &END XC_GRID

       &VDW_POTENTIAL 
         DISPERSION_FUNCTIONAL PAIR_POTENTIAL
         &PAIR_POTENTIAL
           TYPE DFTD3(BJ)
           PARAMETER_FILE_NAME path/to/cp2k-2022.2/data/dftd3.dat
           REFERENCE_FUNCTIONAL PBE
         &END PAIR_POTENTIAL
       &END VDW_POTENTIAL
     &END XC

     &POISSON
       POISSON_SOLVER PERIODIC
       PERIODIC XYZ
     &END POISSON
   &END DFT

   &SUBSYS
     &CELL
       A    21.3948993682999991    0.0000000000000000    0.0000000000000000
       B    0.0000000000000000   21.3948993682999991    0.000000000000000
       C    0.0000000000000000    0.0000000000000000   45.0000000000000000
       PERIODIC  XYZ
       !MULTIPLE_UNIT_CELL  1 1 1
     &END CELL
     &COORD
{coordinates}
      &END COORD
          &KIND O
       BASIS_SET DZVP-MOLOPT-SR-GTH
       POTENTIAL GTH-PBE-q6 
     &END KIND
     &KIND Si
       BASIS_SET DZVP-MOLOPT-SR-GTH
       POTENTIAL GTH-PBE-q4
     &END KIND
     &KIND Al
       BASIS_SET DZVP-MOLOPT-SR-GTH
       POTENTIAL GTH-PBE-q3
     &END KIND
     &KIND Zr
       BASIS_SET TZV2P-MOLOPT-SR-GTH
       POTENTIAL GTH-PBE-q12
     &END KIND
     &KIND H
       BASIS_SET DZVP-MOLOPT-SR-GTH
       POTENTIAL GTH-PBE-q1   
     &END KIND
     &KIND C
       BASIS_SET DZVP-MOLOPT-SR-GTH
       POTENTIAL GTH-PBE-q4
     &END KIND 

     &PRINT
       &TOPOLOGY_INFO
         XYZ_INFO
       &END TOPOLOGY_INFO

       &KINDS
         BASIS_SET
         POTENTIAL
       &END KINDS
     &END PRINT
   &END SUBSYS
&END FORCE_EVAL
"""


def create_submission_script(input_file, output_directory):
    """
    Creates a submission script for a given CP2K input file.
    Args:
        input_file (str): Path to the CP2K input file.
        output_directory (str): Directory to save the submission script.
    """
    os.makedirs(output_directory, exist_ok=True)

    # Extract the base name without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Define the submission script content
    script_content = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -p genoa
#SBATCH -J {base_name}
#SBATCH -t 18:00:00
#$ -cwd

module load 2022
module load CP2K/2022.1-foss-2022a

srun cp2k.popt {base_name}.inp > {base_name}.out
"""

    # Save the script
    script_path = os.path.join(output_directory, f"{base_name}.sh")
    with open(script_path, "w") as script_file:
        script_file.write(script_content)

    print(f"Generated submission script: {script_path}")


def generate_scripts_for_all_inputs(input_directory, output_directory):
    """
    Generates submission scripts for all CP2K input files in a directory.
    Args:
        input_directory (str): Directory containing CP2K input files.
        output_directory (str): Directory to save submission scripts.
    """
    for input_file in os.listdir(input_directory):
        if input_file.endswith(".inp"):
            create_submission_script(os.path.join(input_directory, input_file), output_directory)

if __name__ == "__main__":
    # process the conf files

    conf_directory = "./conf_files/1C"  # Directory containing .conf files
    output_csv = "isomer_analysis_1C.csv"  # Output CSV file for analysis results
    process_multiple_conf_files(conf_directory, output_csv)
    output_csv_longest = "longest_periods_1C.csv"  # Output CSV for longest periods
    find_longest_period(output_csv, output_csv_longest)
    # Directory containing .xyz files
    xyz_directory = "./trajectories/1C"
    # Directory to save CP2K input files
    output_directory = "./cp2k_inputs/1C"
    # Process the longest periods and generate CP2K input files
    process_longest_periods(output_csv_longest, xyz_directory, output_directory, cp2k_template)
    # Generate scripts
    generate_scripts_for_all_inputs(output_directory, output_directory)
