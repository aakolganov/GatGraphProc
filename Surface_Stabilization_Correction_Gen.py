import os.path
from pathlib import Path
from ase import io, Atom
from ase.build import make_supercell
import re
import shutil
from ase import Atoms, Atom


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


def read_dat_file(dat_path):
    """
    Reads the .dat file containing the group information.

    The file is assumed to have one group per line, for example:

       1. 1 4 5 6 7 8 9 10 11 12 13 14 15 17 18 19 20 21 22 23 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 71 76 79 132 133 134 135 ...
       2. 2 3 192 193 194 195 196 197 ...
       3. 16 24 25 27 28 32
       ...

    This function returns two dictionaries:
      - group_map: mapping group_id (int) → list of conformer numbers (int)
      - conf_to_group: mapping conformer number (int) → group_id (int)
    """
    group_map = {}
    conf_to_group = {}
    try:
        with open(dat_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                if not tokens:
                    continue
                group_token = tokens[0].replace('.', '')
                try:
                    group_id = int(group_token)
                except:
                    print(f"Skipping line with invalid group id: {line}")
                    continue
                conformers = []
                for token in tokens[1:]:
                    try:
                        conf_num = int(token)
                        conformers.append(conf_num)
                        conf_to_group[conf_num] = group_id
                    except:
                        continue
                group_map[group_id] = conformers
    except Exception as e:
        print(f"Error reading dat file {dat_path}: {e}")
    return group_map, conf_to_group


def process_xyz_files_with_dat(input_folder, dat_file, output_folder):
    """
    Scans all .xyz files in input_folder with names of the format:
         extracted_group_xxx_frame_yyyy.xyz
    or
         group_xxx_frame_yyyy(-pos-<number>).xyz

    where xxx is the conformer number (1–300) and yyyy is a frame number.

    Reads the .dat file to get the mapping of conformer numbers to group numbers.

    Then, for each file:
      - Extracts the last geometry block (and its energy) from the file.
      - Parses the energy from the comment line.
      - Determines its group from the mapping (conf_to_group).

    For each group (as defined in the .dat file), the file with the lowest energy is
    selected and copied to output_folder with a new name:
         extracted_group_xxx_frame_yyyy_group_zzz.xyz
    where zzz is the group number (taken from the .dat file).
    """
    # Read the .dat file.
    group_map, conf_to_group = read_dat_file(dat_file)
    if not conf_to_group:
        print("No valid group mapping found in the .dat file.")
        return

    # Use a regex that accepts an optional "extracted_" prefix and an optional "-pos-<number>" suffix.
    pattern = re.compile(r'^(?:extracted_)?group_(\d+)_frame_(\d+)(?:-pos-\d+)?\.xyz$')

    # List to hold file info: (filename, conformer, frame, energy, group)
    file_infos = []

    for filename in os.listdir(input_folder):
        if not filename.endswith('.xyz'):
            continue
        match = pattern.fullmatch(filename)
        if not match:
            print(f"Skipping file with unexpected name format: {filename}")
            continue

        conf_str, frame_str = match.groups()
        try:
            conf_num = int(conf_str)
            frame_num = int(frame_str)
        except Exception as e:
            print(f"Error parsing numbers in filename {filename}: {e}")
            continue

        file_path = os.path.join(input_folder, filename)
        atom_count, energy_line, geometry_lines = extract_last_geometry_and_energy(file_path)
        if energy_line is None:
            print(f"Could not extract energy from {filename}")
            continue

        energy = parse_energy_from_line(energy_line)
        if energy is None:
            print(f"Could not parse energy from {filename} using line: {energy_line}")
            continue

        if conf_num not in conf_to_group:
            print(f"Conformer {conf_num} in file {filename} not found in .dat mapping. Skipping.")
            continue
        group_id = conf_to_group[conf_num]
        file_infos.append((filename, conf_num, frame_num, energy, group_id))

    if not file_infos:
        print("No valid .xyz files found.")
        return

    # For each group, select the file with the lowest energy.
    best_files = {}  # key: group_id, value: (filename, conf, frame, energy, group)
    for info in file_infos:
        filename, conf_num, frame_num, energy, group_id = info
        if group_id not in best_files or energy < best_files[group_id][3]:
            best_files[group_id] = info

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Copy each selected file to output_folder with the new naming.
    for group_id, info in sorted(best_files.items(), key=lambda x: x[0]):
        filename, conf_num, frame_num, energy, group_id = info
        src = os.path.join(input_folder, filename)
        new_filename = f"extracted_group_{conf_num}_frame_{frame_num}_group_{group_id}.xyz"
        dst = os.path.join(output_folder, new_filename)
        try:
            shutil.copy2(src, dst)
            print(f"Group {group_id}: Copied {filename} (energy {energy:.10f}) as {new_filename}")
        except Exception as e:
            print(f"Error copying file for group {group_id} from {filename}: {e}")


def extract_and_subst(filename, removable_atoms, SiF3, write_xyz, outfile):
    """
    Extract atoms from an XYZ file, remove specified atoms, adjust Zr-O distance, and generate CP2K input file.

    :param filename: Path to input XYZ file
    :param removable_atoms: atoms to remove
    :param SiF3: Whether to add extra H for the structures w/ SiOH...Al(OC(CF3)3)
    :param write_xyz: Write xyz
    :param outfile: Base name for output files (.inp and optionally .xyz)
    :return: None
    """
    atoms = io.read(filename)
    indexes_to_remove = removable_atoms
    atoms = atoms[[i for i in range(len(atoms)) if i not in indexes_to_remove]]

    # Find the Zr atom and the closest oxygen atom to it, and move the Zr atom to 1.12 Å from the oxygen atom
    zr_index = [i for i, atom in enumerate(atoms) if atom.symbol == 'Zr']
    if zr_index:
        # If there is a Zr atom, find the closest oxygen atom to it
        zr_atom = atoms[zr_index[0]]
        o_indexes = [i for i, atom in enumerate(atoms) if atom.symbol == 'O']
        indices = [zr_index[0]] + o_indexes
        zro_distances = atoms.get_distances(zr_index, indices)
        closest_o_index = o_indexes[zro_distances[1:].argmin()]
        closest_o_atom = atoms[closest_o_index]

        # Move the Zr atom to 1.12 Å from the closest oxygen atom and substitute it with hydrogen
        zr_new_pos = closest_o_atom.position + (zr_atom.position - closest_o_atom.position) / \
            atoms.get_distance(zr_index, closest_o_index) * 0.98
        zr_new_atom = Atom('H', zr_new_pos)
        atoms.pop(zr_index[0])
        atoms.append(zr_new_atom)
    if SiF3:
        new_H_pos = atoms[274].position+[0.68,-0.68, 0.25]
        new_H_atom = Atom('H', new_H_pos)
        atoms.append(new_H_atom)
    name_of_file = os.path.basename(filename)
    file_ident = name_of_file[:-4]
    output_file = outfile[:-4]+'surf_stab'+'.inp'
    with open(output_file, "w", encoding="utf-8") as f:
        gig = (
            "&GLOBAL\n   PRINT_LEVEL  MEDIUM\n   PROJECT_NAME {0} \n   RUN_TYPE  GEO_OPT\n &END GLOBAL\n  \n &MOTION\n   &GEO_OPT\n  MAX_ITER 500 \n    OPTIMIZER BFGS\n     TYPE MINIMIZATION\n   &END GEO_OPT\n &END MOTION  \n\n &FORCE_EVAL\n   METHOD QS\n   &DFT\n     CHARGE 0\n     BASIS_SET_FILE_NAME /gpfs/home4/kolganov/cp2k-2022.2/data/BASIS_MOLOPT_UCL\n     BASIS_SET_FILE_NAME /gpfs/home4/kolganov/cp2k-2022.2/data/BASIS_MOLOPT\n     POTENTIAL_FILE_NAME /gpfs/home4/kolganov/cp2k-2022.2/data/GTH_POTENTIALS \n     &MGRID\n       CUTOFF 450\n       REL_CUTOFF 30\n       NGRIDS 4\n     &END MGRID\n \n     &QS\n       METHOD  GPW\n     &END QS\n             \n     &SCF\n      !MAX_SCF          1             ! Max n of iterations\n       EPS_SCF          0.1E-04       ! SCF converergence\n       SCF_GUESS        ATOMIC\n       &OT\n         MINIMIZER DIIS\n         PRECONDITIONER FULL_SINGLE_INVERSE\n       &END OT\n     &END SCF\n\n     &XC \n       &XC_FUNCTIONAL PBE\n       &END XC_FUNCTIONAL\n      ! &XC_GRID\n       !  XC_DERIV SPLINE2\n       !  XC_SMOOTH_RHO NONE\n      ! &END XC_GRID\n\n       &VDW_POTENTIAL \n         DISPERSION_FUNCTIONAL PAIR_POTENTIAL\n         &PAIR_POTENTIAL\n           TYPE DFTD3(BJ)\n           PARAMETER_FILE_NAME /gpfs/home4/kolganov/cp2k-2022.2/data/dftd3.dat\n           REFERENCE_FUNCTIONAL PBE\n         &END PAIR_POTENTIAL\n       &END VDW_POTENTIAL\n     &END XC\n     \n     &POISSON\n       POISSON_SOLVER PERIODIC\n       PERIODIC XYZ\n     &END POISSON\n   &END DFT\n\n   &SUBSYS\n     &CELL\n       A    21.3948993682999991    0.0000000000000000    0.0000000000000000\n       B    0.0000000000000000   21.3948993682999991    0.000000000000000\n       C    0.00000000000000000    0.0000000000000000   45.0000000000000000\n       PERIODIC  XYZ\n       !MULTIPLE_UNIT_CELL  1 1 1\n     &END CELL\n     &COORD\n").format(
            file_ident)
        f.write(gig)
        for atom in atoms:
            f.write(f"{atom.symbol} {atom.x} {atom.y} {atom.z}\n")
        f.write(
            "&END COORD\n          \n     &KIND O\n       BASIS_SET DZVP-MOLOPT-SR-GTH\n       POTENTIAL GTH-PBE-q6 \n     &END KIND\n     &KIND Si\n       BASIS_SET DZVP-MOLOPT-SR-GTH\n       POTENTIAL GTH-PBE-q4\n     &END KIND\n     &KIND Al\n       BASIS_SET DZVP-MOLOPT-SR-GTH\n       POTENTIAL GTH-PBE-q3\n     &END KIND\n     &KIND Zr\n       BASIS_SET TZV2P-MOLOPT-SR-GTH\n       POTENTIAL GTH-PBE-q12\n     &END KIND\n     &KIND H\n             BASIS_SET DZVP-MOLOPT-SR-GTH\n       POTENTIAL GTH-PBE-q1    \n     &END KIND\n     &KIND C\n       BASIS_SET DZVP-MOLOPT-SR-GTH\n       POTENTIAL GTH-PBE-q4\n     &END KIND\n &KIND F\n       BASIS_SET DZVP-MOLOPT-SR-GTH\n       POTENTIAL GTH-PBE-q7\n     &END KIND \n      &KIND S \n   BASIS_SET DZVP-MOLOPT-SR-GTH \n  POTENTIAL GTH-PBE-q6 \n &END KIND  \n   \n     &PRINT\n       &TOPOLOGY_INFO\n         XYZ_INFO\n       &END TOPOLOGY_INFO\n      \n       &KINDS\n         BASIS_SET\n         POTENTIAL\n       &END KINDS\n     &END PRINT\n         \n   &END SUBSYS\n &END FORCE_EVAL\n\n")
    f.close()
    if write_xyz:
        output_xyz = outfile[:-4]+'surf_stab'+'.xyz'
        io.write(output_xyz, atoms)




#example usage
if __name__ == '__main__':
    # Set your folder paths:
    input_folder = 'xyz_opted/1C_allxyz/'  # Folder containing extracted_group_xxx_frame_yyyy.xyz files
    dat_file = 'Post_proc_single_Gateway/1C_ZrC_short/processed_concatenated_geometries_periods_1C.dat'            # The .dat file with group mapping
    output_folder = 'Post_proc_single_Gateway/1C_ZrC_short/Surf_Rel'      # Where the lowest-energy representatives will be copied

    process_xyz_files_with_dat(input_folder, dat_file, output_folder)

    directory = '/Users/akolganov/PycharmProjects/ZrCp2_automatizing/Post_proc_single_Gateway/1C_ZrC_short/Surf_Rel'
    indexes_to_remove_first_row = [i for i in range(35, 62)]
    indexes_to_remove_second_row = [i for i in range(62, 80)]
    indexes_to_remove_Zr = [459]
    indexes_to_remove = indexes_to_remove_first_row + indexes_to_remove_second_row + indexes_to_remove_Zr

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('xyz'):
                file_path = os.path.join(root, file)
                # Define the output directory (note: no leading slash in the folder name)
                new_subdir = os.path.join(directory, 'surf_stab')
                # Create the subdirectory if it does not exist
                os.makedirs(new_subdir, exist_ok=True)
                new_file = os.path.join(new_subdir, file)
                extract_and_subst(file_path, indexes_to_remove, False, True, new_file)