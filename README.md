Simple workflow to process and visualize the GaTeway graph theory analysis outputs

GaTeway paper: https://chemrxiv.org/engage/chemrxiv/article-details/63590093ecdad54aaaed9777

This workflow is highlighted in this publication: 

https://pubs.rsc.org/en/content/articlelanding/2025/cp/d4cp02764g

The workflow works in the following order:

1. Generate MD data
2. Process it with GaTeway
3. Execute Gateway_stage1_processing.py to extract unique configurations and prepare input files for CP2K
3.5 (Optional) - Visualize 1st gen graphs with Draw_Stage1_Transition_Graphs.py
4. Optimize these isomers with CP2K and compile them into one "trajectory" to process them with GaTeway again.
5. Process and visualize second-gen graphs with Draw_Stage2_Transition_Graphs.py
