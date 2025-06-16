import os

import numpy
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import manifold, datasets
import sklearn.decomposition
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn_extra.cluster import KMedoids
from sklearn import metrics
import seaborn as sns
import plotly.graph_objs as go
import itertools
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import MDAnalysis as mda
from MDAnalysis.analysis import align

directory =''

def extract_md_traj(ref_file, cuwodir, ends_with, atomindexes, number_of_atoms):
    """
    Extracts and processes molecular dynamics trajectory data from simulation files.

    Parameters
    ----------
    ref_file : str
        Path to the reference structure file.
    cuwodir : str
        Directory containing the trajectory files to process.
    ends_with : str
        File extension to filter trajectory files.
    atomindexes : str
        Atom selection string for alignment and RMSD calculation.
    number_of_atoms : int
        Number of atoms in the selection for reshaping coordinates.

    Returns
    -------
    tuple
        Contains two numpy arrays:
        - dyn_zr : numpy.ndarray
            Reshaped coordinate array of selected atoms (frames × (number_of_atoms * 3))
        - rmsd : numpy.ndarray
            Array of RMSD values with corresponding file names and frame numbers
    """
    ref = mda.Universe(ref_file)
    rmsd = []  #rmsd
    all_dyn = []  # all
    all_dyn_no_al = []  #all, no aligр
    dyn_zr = []  #only_Zr
    n_steps = 50  #extracting each step
    for subdir, dirs, files in os.walk(cuwodir):
        for file in files:
            if file.endswith(ends_with):
                traj = mda.Universe(os.path.join(directory, file))  #extracting frames from MD
                traj.transfer_to_memory(start=0, step=n_steps, stop=10000)
                for ts in traj.trajectory:
                    all_dyn.append(traj.select_atoms("all").ts.positions)  #no_aligment_append
                    alignment = align.alignto(traj, ref, select=atomindexes)  #aligment
                    rmsd.append([file, (traj.select_atoms(atomindexes).ts.frame * n_steps), \
                                 alignment[1]])  # rmsd
                    dyn_zr.append(traj.select_atoms(atomindexes).ts.positions)  #key dynamics coords
                    all_dyn_no_al.append(traj.select_atoms("all").ts.positions)
    dyn_zr = np.array(dyn_zr).reshape(int(10000 / n_steps), (number_of_atoms) * 3)
    rmsd = np.array(rmsd)
    return (dyn_zr, rmsd)


def extract_md_int_coords(ref_file, cuwodir, ends_with, atomindexes):
    """
    Extracts internal coordinates from molecular dynamics trajectory files and calculates distances.

    Parameters
    ----------
    ref_file : str
        Path to the reference structure file.
    cuwodir : str
        Directory containing the trajectory files to process.
    ends_with : str
        File extension to filter trajectory files.
    atomindexes : str
        Atom selection string for alignment and coordinate extraction.

    Returns
    -------
    numpy.ndarray
        Array of distances between selected atoms and the last atom's coordinates
        for each frame in the trajectory.
    """
    ref = mda.Universe(ref_file)
    dyn_zr = []  #only_Zr
    n_steps = 50
    for subdir, dirs, files in os.walk(cuwodir):
        for file in files:
            if file.endswith(ends_with):
                traj = mda.Universe(os.path.join(directory, file))
                traj.transfer_to_memory(start=0, step=n_steps, stop=10000)
                for ts in traj.trajectory:
                    alignment = align.alignto(traj, ref, select=atomindexes)  #aligment
                    dyn_zr.append(traj.select_atoms(atomindexes).ts.positions)
    dyn_zr = np.array(dyn_zr)
    last_atom_coords = dyn_zr[:, -1, :]
    distances = np.linalg.norm(dyn_zr - last_atom_coords[:, np.newaxis, :], axis=2)
    return (distances)

def clustering_test_tsne(dynamics, bottom_clust_num, top_clust_num, dyn_name):
    """
    Performs t-SNE dimensionality reduction and K-Medoids clustering analysis on molecular dynamics data.

    Parameters
    ----------
    dynamics : numpy.ndarray
        Input array containing molecular dynamics trajectory data.
    bottom_clust_num : int
        Minimum number of clusters to test.
    top_clust_num : int
        Maximum number of clusters to test.
    dyn_name : str
        Name of the dynamics dataset for plot titles and file names.

    Returns
    -------
    numpy.ndarray
        Indices of medoid points for the optimal clustering solution.
    """

    sse = []
    tsne = manifold.TSNE(n_components=2, perplexity=9)
    tsne_ = tsne.fit_transform(dynamics)
    for i in range(bottom_clust_num, top_clust_num):
        clustering = KMedoids(n_clusters=i, max_iter=5000, random_state=346, method='pam', init='build').fit( ## k-medoids++
            tsne_)
        silhouette_avg = silhouette_score(dynamics, clustering.labels_, metric="euclidean")
        sse.append([i, clustering.inertia_, silhouette_avg])
    sse = np.array(sse)
    fig, axs6 = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
    title1 = 'Optimal Number of Clusters \n using Elbow Method \n for {0}'.format(dyn_name)
    axs6[0].plot(sse[:, 0], sse[:, 1], markersize=12, color='skyblue', linewidth=4)
    axs6[0].set_title(title1)
    axs6[0].set_xlabel('Number of clusters')
    axs6[0].set_ylabel('Inertia')
    title2 = 'Optimal Number of Clusters \n using Silhouette Method (t-SNE) \n for {0}'.format(dyn_name)
    axs6[1].plot(sse[:, 0], sse[:, 2], markersize=12, color='gold', linewidth=4)
    axs6[1].set_title(title2)
    axs6[1].set_xlabel('Number of clusters')
    axs6[1].set_ylabel('Silhouette Score')
    plt_filename = "clustering_test" + dyn_name[:-4] + ".png"
    plt_name = os.path.join(directory, plt_filename)
    plt.savefig(plt_name, dpi = 'figure')
    plt.show()
    sil_max = np.where(sse[:, 2] == max(sse[:, 2]))
    clustering_final = KMedoids(n_clusters=int(sse[sil_max, :1]), random_state=15).fit(dynamics)
    plt.figure(figsize=(5, 5))
    title2 = 'KMeans Clusters Derived from Original Dataset \n t-SNE DIM. Red. \n {}'.format(dyn_name)
    sns.scatterplot(x=tsne_[:, 0], y=tsne_[:, 1],
                    hue=clustering_final.labels_, markers="o", s=100,
                    palette='deep').set_title(
        title2, fontsize=10)
    sns.scatterplot(x=tsne_[clustering_final.medoid_indices_][:, 0], y=tsne_[clustering_final.medoid_indices_][:, 1],
                    marker="o", \
                    zorder=100, linewidth=2, color="black")
    plt.legend(loc=(1.15, 0), ncol=7)
    plt.ylabel('t-SNE2')
    plt.xlabel('t-SNE1')
    plt_filename_1 = "final_clust" + dyn_name[:-4] + ".png"
    plt_name_1 = os.path.join(directory, plt_filename_1)
    plt.savefig(plt_name_1, dpi = 'figure')
    plt.show()
    return (clustering_final.medoid_indices_)

def clusters_to_CP2K_inps(indexes_md, cuwodir, ends_with, xyz):
    """
    Generates CP2K input files for selected frames from molecular dynamics trajectories.

    Parameters
    ----------
    indexes_md : numpy.ndarray
        Array of frame indices selected from clustering analysis.
    cuwodir : str
        Directory containing the trajectory files to process.
    ends_with : str
        File extension to filter trajectory files.
    xyz : bool
        If True, also generates XYZ coordinate files for selected frames.

    Returns
    -------
    None
        Creates input files and optionally XYZ files in a new directory.
    """

    final_list = []
    for i in indexes_md:
        c = i * 50
        final_list.append(c)
    for subdir, dirs, files in os.walk(cuwodir):
        for file in files:
            if file.endswith(ends_with):
                filename = os.path.join(cuwodir, file)
                with open(filename, "r") as f:
                    lines = f.readlines()
                    f.close()
    num_atoms = int(lines[0])
    num_lines_per_frame = num_atoms + 2
    output_dir = cuwodir + '_output_' + ends_with[:-3]
    os.makedirs(output_dir)
    for i in final_list:
        frame_lines = lines[i * num_lines_per_frame + 2: (i + 1) * num_lines_per_frame]
        output_filename = os.path.join(output_dir, f"frame_{i}.inp")
        with open(output_filename, "w") as f:
            gig = (
                "&GLOBAL\n   PRINT_LEVEL  MEDIUM\n   PROJECT_NAME {0}_frame \n   RUN_TYPE  GEO_OPT\n &END GLOBAL\n  \n &MOTION\n   &GEO_OPT\n  MAX_ITER 500 \n    OPTIMIZER BFGS\n     TYPE MINIMIZATION\n   &END GEO_OPT\n &END MOTION  \n\n &FORCE_EVAL\n   METHOD QS\n   &DFT\n     CHARGE 0\n     BASIS_SET_FILE_NAME /gpfs/home4/kolganov/cp2k-2022.2/data/BASIS_MOLOPT_UCL\n     BASIS_SET_FILE_NAME /gpfs/home4/kolganov/cp2k-2022.2/data/BASIS_MOLOPT\n     POTENTIAL_FILE_NAME /gpfs/home4/kolganov/cp2k-2022.2/data/GTH_POTENTIALS \n     &MGRID\n       CUTOFF 450\n       REL_CUTOFF 30\n       NGRIDS 4\n     &END MGRID\n \n     &QS\n       METHOD  GPW\n     &END QS\n             \n     &SCF\n      !MAX_SCF          1             ! Max n of iterations\n       EPS_SCF          0.1E-04       ! SCF converergence\n       SCF_GUESS        ATOMIC\n       &OT\n         MINIMIZER DIIS\n         PRECONDITIONER FULL_SINGLE_INVERSE\n       &END OT\n     &END SCF\n\n     &XC \n       &XC_FUNCTIONAL PBE\n       &END XC_FUNCTIONAL\n      ! &XC_GRID\n       !  XC_DERIV SPLINE2\n       !  XC_SMOOTH_RHO NONE\n      ! &END XC_GRID\n\n       &VDW_POTENTIAL \n         DISPERSION_FUNCTIONAL PAIR_POTENTIAL\n         &PAIR_POTENTIAL\n           TYPE DFTD3(BJ)\n           PARAMETER_FILE_NAME /gpfs/home4/kolganov/cp2k-2022.2/data/dftd3.dat\n           REFERENCE_FUNCTIONAL PBE\n         &END PAIR_POTENTIAL\n       &END VDW_POTENTIAL\n     &END XC\n     \n     &POISSON\n       POISSON_SOLVER PERIODIC\n       PERIODIC XYZ\n     &END POISSON\n   &END DFT\n\n   &SUBSYS\n     &CELL\n       A    21.3948993682999991    0.0000000000000000    0.0000000000000000\n       B    0.0000000000000000   21.3948993682999991    0.000000000000000\n       C    0.00000000000000000    0.0000000000000000   45.0000000000000000\n       PERIODIC  XYZ\n       !MULTIPLE_UNIT_CELL  1 1 1\n     &END CELL\n     &COORD\n").format(
                str(i)+ends_with)
            f.write(gig)
            f.writelines(frame_lines)
            f.write(
                "&END COORD\n          \n     &KIND O\n       BASIS_SET DZVP-MOLOPT-SR-GTH\n       POTENTIAL GTH-PBE-q6 \n     &END KIND\n     &KIND Si\n       BASIS_SET DZVP-MOLOPT-SR-GTH\n       POTENTIAL GTH-PBE-q4\n     &END KIND\n     &KIND Al\n       BASIS_SET DZVP-MOLOPT-SR-GTH\n       POTENTIAL GTH-PBE-q3\n     &END KIND\n     &KIND Zr\n       BASIS_SET TZV2P-MOLOPT-SR-GTH\n       POTENTIAL GTH-PBE-q12\n     &END KIND\n     &KIND H\n             BASIS_SET DZVP-MOLOPT-SR-GTH\n       POTENTIAL GTH-PBE-q1    \n     &END KIND\n     &KIND C\n       BASIS_SET DZVP-MOLOPT-SR-GTH\n       POTENTIAL GTH-PBE-q4\n     &END KIND\n &KIND F\n       BASIS_SET DZVP-MOLOPT-SR-GTH\n       POTENTIAL GTH-PBE-q7\n     &END KIND \n      &KIND S \n   BASIS_SET DZVP-MOLOPT-SR-GTH \n  POTENTIAL GTH-PBE-q6 \n &END KIND  \n   \n     &PRINT\n       &TOPOLOGY_INFO\n         XYZ_INFO\n       &END TOPOLOGY_INFO\n      \n       &KINDS\n         BASIS_SET\n         POTENTIAL\n       &END KINDS\n     &END PRINT\n         \n   &END SUBSYS\n &END FORCE_EVAL\n\n")
        f.close()
    if xyz:
            for i in final_list:
                frame_lines = lines[i * num_lines_per_frame + 2: (i + 1) * num_lines_per_frame]
                output_filename = os.path.join(output_dir, f"frame_{i}.xyz")
                with open(output_filename, "w") as f:
                    f.writelines(frame_lines)
                    f.close()

if __name__ == "__main__":
    directory = "C:\\Users\\akolganov\\OneDrive\\Рабочий стол\\Манускрипты\\Cp2Zr_Al2O3\\CP2K_systematize_Dynamics\\Bare_Silica\\Initial_complex\\All_dynamic_files\\"

    reference = os.path.join(directory, "expert.xyz")
    #Clustering PhysAds dataaaaa

    file_end_list = ["353.xyz", "773.xyz", "MD_1.xyz", "MD_2.xyz", "MD_3.xyz"]

    file_end_test = ["MD_1.xyz"]

    indexes_all = "all"  #419 atoms
    indexes_all_C_atoms_in_AS = "index 38:38, 63:81, 418:418"  #21 atoms
    indexes_Bu_and_ZrH2_atoms_in_AS = "index 38:38, 63:63, 74:81, 418:418"  #11 atoms
    indexes_only_Zr_and_H = "index 38:38, 63:63, 418:418"  #3 atoms
    indexes_ZrH2_and_SiOH = "index 38:38, 63:63, 418:418, 94:94, 335:335, 62:62"  #6 atoms
    indexes_Bu_and_ZrH2_and_SiOH = "index 38:38, 63:63, 74:81, 418:418, 94:94, 335:335, 62:62"  #14 atoms

    rmsds = []
    dyns = []
    indexes_md = []
    dists = []
    #
    # for name in file_end_list:
    #     dyn = extract_md_int_coords(reference, directory, name, indexes_only_Zr_and_H)
    #     dists.append(dyn)
    #     sse = clustering_test(dyn, 2, 100, name)
    #     indexes_md.append(final_clustering(dyn, sse, name))

    #


    for name in file_end_list:
         dyn, rmsd = extract_md_traj(reference, directory, name, indexes_Bu_and_ZrH2_and_SiOH, 14)
         rmsd = np.array(rmsd)
         rmsds.append(rmsd)
         dyns.append(dyn)
         #indexes_md = clustering_test_PCA(dyn, 2, 50, name)
         indexes_md = clustering_test_tsne(dyn, 2, 50, name)
         #clusters_to_CP2K_inps(indexes_md, directory, name, False)

    # RMSD_plot
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.plot(rmsds[0][:, 1].astype(float), rmsds[0][:, 2].astype(float),
            linewidth=1, color="darkorange", label="ai-353K")
    ax.plot(rmsds[1][:, 1].astype(float), rmsds[1][:, 2].astype(float),
            linewidth=1, color="blue", label="ai-773K")
    ax.plot(rmsds[2][:, 1].astype(float), rmsds[2][:, 2].astype(float),
            linewidth=1, color="red", label="LM1")
    ax.plot(rmsds[3][:, 1].astype(float), rmsds[3][:, 2].astype(float),
            linewidth=1, color="black", label="LM2")
    ax.plot(rmsds[3][:, 1].astype(float), rmsds[4][:, 2].astype(float),
            linewidth=1, color="green", label="LM3")

    ax.legend(("ai-353", "ai-773", "VS1", "VS2", "VS3"), loc="best")
    ax.set_xlabel("frame")
    plt.xlim((0, 10000))
    plt.ylim((0, 10))
    ax.set_ylabel(r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ ($\AA$)")
    ax.set_ylabel(r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ ($\AA$)")
    plt_name = os.path.join(directory, 'rmsd_info')
    sns.set_style('white')
    plt.grid(False)
    plt.savefig(plt_name, dpi='figure')


